from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from comfy.model_patcher import ModelPatcher


@dataclass
class QuantizationRunStats:
    converted: int = 0
    kept: int = 0
    skipped_no_stats: int = 0
    skipped_already_fp8: int = 0
    failed: int = 0


def resolve_stats_path(path: str) -> str:
    if os.path.exists(path):
        return path
    try:
        import folder_paths

        alt = os.path.join(folder_paths.get_output_directory(), path)
        if os.path.exists(alt):
            return alt
    except Exception:
        pass
    return path


def encode_comfy_quant(fmt: str = "float8_e4m3fn") -> torch.Tensor:
    return torch.tensor(list(json.dumps({"format": fmt}).encode("utf-8")), dtype=torch.uint8)


def del_buffer(module: nn.Module, name: str) -> None:
    if hasattr(module, "_buffers") and name in module._buffers:
        del module._buffers[name]


def clear_model_wrapper(model) -> object:
    work_model = model.clone()
    if hasattr(work_model, "set_model_unet_function_wrapper"):
        try:
            work_model.set_model_unet_function_wrapper(None)
        except Exception:
            pass
    return work_model


def get_diffusion_model(work_model):
    if isinstance(work_model, ModelPatcher):
        return work_model.model.diffusion_model
    return work_model.diffusion_model


def load_session_data(stats_path: str, *, log_prefix: str):
    resolved_path = resolve_stats_path(stats_path)
    if not os.path.exists(resolved_path):
        print(f"[{log_prefix}] Error: Stats file not found: {resolved_path}")
        return resolved_path, None

    try:
        session_data = torch.load(resolved_path, map_location="cpu")
    except Exception as exc:
        print(f"[{log_prefix}] Error loading stats: {exc}")
        return resolved_path, None

    meta = session_data.get("meta", {})
    if meta.get("type") != "hswq_dual_monitor_v2":
        print(f"[{log_prefix}] Warning: meta.type is '{meta.get('type')}', expected 'hswq_dual_monitor_v2'.")

    layers_data = session_data.get("layers", {})
    if not layers_data:
        print(f"[{log_prefix}] Error: No layers found in stats.")
        return resolved_path, None

    return resolved_path, session_data


def calculate_variance(stats: dict) -> float:
    count = int(stats.get("out_count", 0))
    if count <= 0:
        return 0.0
    mean = stats["output_sum"] / count
    sq_mean = stats["output_sq_sum"] / count
    variance = sq_mean - (mean ** 2)
    return float(max(variance, 0.0))


def calculate_sensitivities(layers_data: dict[str, dict]) -> list[tuple[str, float]]:
    sensitivities = [(name, calculate_variance(stats)) for name, stats in layers_data.items() if int(stats.get("out_count", 0)) > 0]
    sensitivities.sort(key=lambda item: item[1], reverse=True)
    return sensitivities


def build_keep_names_from_sensitivities(sensitivities: list[tuple[str, float]], keep_ratio: float) -> tuple[set[str], int]:
    num_keep = int(len(sensitivities) * float(keep_ratio))
    return {name for name, _ in sensitivities[:num_keep]}, num_keep


def quantize_diffusion_model(
    diffusion_model,
    *,
    layers_data: dict,
    keep_names: set[str],
    optimizer,
    fp8_quantizer,
    device,
    scaled: bool,
    inject_comfy_metadata: bool,
    log_level: str,
    log_prefix: str,
    amax_adjuster: Callable[[str, torch.Tensor, float], float] | None = None,
) -> QuantizationRunStats:
    meta_proto = encode_comfy_quant("float8_e4m3fn")
    fp8_max = float(getattr(fp8_quantizer, "max_representable", 448.0))
    run_stats = QuantizationRunStats()

    for name, module in diffusion_model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if name not in layers_data:
            run_stats.skipped_no_stats += 1
            continue
        if module.weight.dtype == torch.float8_e4m3fn:
            run_stats.skipped_already_fp8 += 1
            continue
        if name in keep_names:
            run_stats.kept += 1
            del_buffer(module, "comfy_quant")
            del_buffer(module, "weight_scale")
            if module.weight.dtype == torch.bfloat16:
                module.weight.data = module.weight.data.to(torch.float16)
            if module.bias is not None and module.bias.dtype == torch.bfloat16:
                module.bias.data = module.bias.data.to(torch.float16)
            continue

        layer_stats = layers_data[name]
        in_count = int(layer_stats.get("in_count", 0))
        importance = None
        if in_count > 0 and isinstance(layer_stats.get("input_imp_sum"), torch.Tensor):
            importance = (layer_stats["input_imp_sum"] / in_count).float()

        try:
            weight = module.weight.data.detach()
            amax = float(optimizer.compute_optimal_amax(weight, importance, scaled=scaled))
            if amax_adjuster is not None:
                amax = float(amax_adjuster(name, weight, amax))
            if not (amax > 0):
                run_stats.failed += 1
                continue

            weight_device = weight.to(device=device, dtype=torch.float16)
            if scaled:
                scale = fp8_max / max(amax, 1e-12)
                weight_fp8 = (weight_device * scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                weight_scale = amax / fp8_max
            else:
                clip = min(amax, fp8_max)
                weight_fp8 = weight_device.clamp(-clip, clip).to(torch.float8_e4m3fn)
                weight_scale = 1.0

            if not torch.isfinite(weight_fp8.float()).all():
                if log_level in ["Verbose", "Debug"]:
                    print(f"[{log_prefix}] Reject (non-finite) -> keep FP16: {name}")
                run_stats.failed += 1
                continue

            module.weight.data = weight_fp8.to(weight.device)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float16)

            if inject_comfy_metadata:
                del_buffer(module, "comfy_quant")
                del_buffer(module, "weight_scale")
                module.register_buffer("comfy_quant", meta_proto.clone().to(weight.device))
                module.register_buffer(
                    "weight_scale",
                    torch.tensor(float(weight_scale), dtype=torch.float32, device=weight.device),
                )
            else:
                del_buffer(module, "comfy_quant")
                del_buffer(module, "weight_scale")

            run_stats.converted += 1
            if log_level == "Debug":
                print(f"[{log_prefix}] {name}: amax={amax:.6g}, scaled={scaled}, weight_scale={weight_scale:.6g}")
        except Exception as exc:
            run_stats.failed += 1
            if log_level in ["Verbose", "Debug"]:
                import traceback

                print(f"[{log_prefix}] Failed: {name} -> {exc}")
                traceback.print_exc()

    return run_stats
