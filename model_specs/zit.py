from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..core.quantization_common import build_keep_names_from_sensitivities, calculate_sensitivities
from ..core.session_store import build_default_meta


HOOK_ATTR = "_zit_hswq_calibration_hooks"
CALIBRATION_LOG_PREFIX = "ZITCollector"
QUANTIZATION_LOG_PREFIX = "ZIT-HSWQ"
OPTIMIZER_MODE = "fast"
TARGET_LAYER_OPTIONS = ["all_linear_conv", "attention_only", "feed_forward_only", "context_refiner"]
ZIT_PREFIXES = [
    "model.diffusion_model.",
    "model.",
    "diffusion_model.",
    "",
]


def create_session_meta() -> dict:
    return build_default_meta(model_type="NextDiT")


def should_hook_layer(layer_name: str, target_layer: str) -> bool:
    if target_layer == "all_linear_conv":
        return True
    if target_layer == "attention_only":
        return ("attn" in layer_name) or ("qkv" in layer_name)
    if target_layer == "feed_forward_only":
        return ("feed_forward" in layer_name) or ("ffn" in layer_name)
    if target_layer == "context_refiner":
        return "context_refiner" in layer_name
    return True


def calculate_kurtosis(tensor: torch.Tensor) -> float:
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0:
        return 0.0
    return torch.mean(((tensor - mean) / std) ** 4).item()


def normalize_profile_keys(model_profile: dict[str, dict]) -> dict[str, dict]:
    if not model_profile:
        return {}

    sample_key = next(iter(model_profile))
    profile_prefix = ""
    for prefix in ZIT_PREFIXES:
        if prefix and sample_key.startswith(prefix):
            profile_prefix = prefix
            break

    normalized = {}
    for key, value in model_profile.items():
        stripped = key[len(profile_prefix):] if profile_prefix and key.startswith(profile_prefix) else key
        normalized[stripped] = value
    return normalized


def load_profile_data(profile_path: str) -> dict[str, dict]:
    if not profile_path:
        return {}

    resolved_path = profile_path
    if not os.path.exists(resolved_path):
        try:
            import folder_paths

            alt = os.path.join(folder_paths.get_output_directory(), profile_path)
            if os.path.exists(alt):
                resolved_path = alt
        except Exception:
            pass

    if not os.path.exists(resolved_path):
        return {}

    with open(resolved_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return normalize_profile_keys(data.get("layers", data))


def derive_hswq_strategy(model_profile: dict[str, dict]):
    model_profile = normalize_profile_keys(model_profile)

    def get_dynamic_search_low(name: str, weight_tensor: torch.Tensor) -> float:
        profile = model_profile.get(name + ".weight", model_profile.get(name, {})) if model_profile else {}
        if profile:
            kurtosis = profile.get("kurtosis", 0)
            outlier_ratio = profile.get("outlier_ratio", 0)
        else:
            weight_f32 = weight_tensor.float()
            kurtosis = calculate_kurtosis(weight_f32)
            std = torch.std(weight_f32).item()
            abs_max = max(abs(weight_f32.min().item()), abs(weight_f32.max().item()))
            outlier_ratio = float(abs_max / std if std > 0 else 0)

        kurtosis_penalty = min(kurtosis / 100.0, 0.49)
        outlier_penalty = min(outlier_ratio / 60.0, 0.49)
        return float(np.clip(0.50 + max(kurtosis_penalty, outlier_penalty), 0.50, 0.99))

    if not model_profile:
        return 0.5, 0.5, get_dynamic_search_low, set()

    kurtosis_values = [value.get("kurtosis", 0) for value in model_profile.values() if isinstance(value, dict)]
    avg_kurtosis = sum(kurtosis_values) / len(kurtosis_values) if kurtosis_values else 0
    kurtosis_factor = min(avg_kurtosis / 50.0, 0.3)
    alpha = float(np.clip(0.5 + kurtosis_factor, 0.5, 0.8))
    beta = 1.0 - alpha

    hard_veto_layers = set()
    for name, profile in model_profile.items():
        if not isinstance(profile, dict):
            continue
        kurtosis = profile.get("kurtosis", 0)
        abs_max = profile.get("abs_max", 0)
        outlier_ratio = profile.get("outlier_ratio", 0)
        is_extreme_divergence = outlier_ratio > 40
        is_extreme_kurtosis = kurtosis > 20
        is_huge_magnitude = abs_max > 20
        if is_extreme_divergence or is_extreme_kurtosis or is_huge_magnitude:
            hard_veto_layers.add(name[:-7] if name.endswith(".weight") else name)

    return alpha, beta, get_dynamic_search_low, hard_veto_layers


def build_selection_plan(layers_data: dict, keep_ratio: float, profile_data: dict[str, dict] | None = None) -> dict:
    profile_data = normalize_profile_keys(profile_data or {})
    alpha, beta, get_layer_search_low, hard_veto_layers = derive_hswq_strategy(profile_data)
    normalized_profile = {}
    for key, value in profile_data.items():
        stripped = key[:-7] if key.endswith(".weight") else key
        normalized_profile[stripped] = value

    layer_scores = []
    raw_sensitivities = dict(calculate_sensitivities(layers_data))
    for name in layers_data.keys():
        if name in hard_veto_layers:
            continue

        sensitivity = raw_sensitivities.get(name, 0.0)
        profile = normalized_profile.get(name, {})
        if profile:
            kurtosis = profile.get("kurtosis", 0)
            outlier_ratio = profile.get("outlier_ratio", 0)
            abs_max = profile.get("abs_max", 0)
            profile_score = (kurtosis * 1.5) + (outlier_ratio * 2.0) + (abs_max * 0.5)
        else:
            profile_score = 0.0

        final_score = (alpha * sensitivity) + (beta * profile_score)
        layer_scores.append((name, final_score))

    layer_scores.sort(key=lambda item: item[1], reverse=True)
    dynamic_keep_names, num_keep = build_keep_names_from_sensitivities(layer_scores, keep_ratio)
    keep_names = dynamic_keep_names.union(hard_veto_layers)

    def amax_adjuster(name: str, weight_tensor: torch.Tensor, amax: float) -> float:
        search_low = get_layer_search_low(name, weight_tensor)
        min_allowed_amax = weight_tensor.abs().max().item() * search_low
        return max(amax, min_allowed_amax)

    extra_logs = [
        f"Dynamic kept (from non-VETO pool): {len(dynamic_keep_names)} (Top {keep_ratio * 100:.1f}% of {len(layer_scores)})",
        f"Static kept (Hard VETO): {len(hard_veto_layers)} (Always FP16)",
        f"Final FP16 kept layers: {len(keep_names)} (VETO {len(hard_veto_layers)} + Dynamic {len(dynamic_keep_names)})",
    ]
    if hard_veto_layers:
        extra_logs.append("Hard VETO layers: " + ", ".join(sorted(hard_veto_layers)))

    return {
        "sensitivities": layer_scores,
        "keep_names": keep_names,
        "num_keep": num_keep,
        "extra_logs": extra_logs,
        "amax_adjuster": amax_adjuster,
        "hard_veto_layers": hard_veto_layers,
        "dynamic_keep_names": dynamic_keep_names,
    }
