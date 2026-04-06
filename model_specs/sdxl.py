from __future__ import annotations

from ..core.quantization_common import build_keep_names_from_sensitivities, calculate_sensitivities
from ..core.session_store import build_default_meta


HOOK_ATTR = "_hswq_calibration_hooks"
CALIBRATION_LOG_PREFIX = "HSWQCollector"
QUANTIZATION_LOG_PREFIX = "HSWQ"
OPTIMIZER_MODE = "fast"


def create_session_meta() -> dict:
    return build_default_meta()


def should_hook_layer(_layer_name: str) -> bool:
    return True


def build_selection_plan(layers_data: dict, keep_ratio: float) -> dict:
    sensitivities = calculate_sensitivities(layers_data)
    keep_names, num_keep = build_keep_names_from_sensitivities(sensitivities, keep_ratio)
    return {
        "sensitivities": sensitivities,
        "keep_names": keep_names,
        "num_keep": num_keep,
        "extra_logs": [],
        "amax_adjuster": None,
    }

