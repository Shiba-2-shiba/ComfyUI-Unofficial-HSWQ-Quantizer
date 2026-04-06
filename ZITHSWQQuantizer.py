import torch

import comfy.model_management

from .core.optimizer_bridge import create_fp8_quantizer, create_optimizer
from .core.quantization_common import (
    clear_model_wrapper,
    del_buffer as _del_buffer_impl,
    encode_comfy_quant as _encode_comfy_quant_impl,
    get_diffusion_model,
    load_session_data,
    quantize_diffusion_model,
    resolve_stats_path as _resolve_stats_path_impl,
)
from .hswq_comfy_api import IO
from .model_specs import zit as zit_spec


def _resolve_stats_path(path: str) -> str:
    return _resolve_stats_path_impl(path)


def _encode_comfy_quant(fmt: str = "float8_e4m3fn") -> torch.Tensor:
    return _encode_comfy_quant_impl(fmt)


def _del_buffer(module, name: str):
    _del_buffer_impl(module, name)


class ZITHSWQQuantizerNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ZITHSWQQuantizerNode",
            display_name="ZIT HSWQ FP8 Quantizer (Spec-aligned)",
            category="ZIT/Quantization",
            description="Quantize a ZIT/NextDiT model to FP8 using HSWQ calibration stats.",
            inputs=[
                IO.Model.Input("model"),
                IO.String.Input("stats_path", default="output/zit_hswq_stats/zit_calib_session_01.pt"),
                IO.String.Input("profile_path", default=""),
                IO.Float.Input("keep_ratio", default=0.25, min=0.0, max=1.0, step=0.05),
                IO.Int.Input("bins", default=8192, min=512, max=65536, step=512),
                IO.Int.Input("num_candidates", default=1000, min=50, max=5000, step=50),
                IO.Int.Input("refinement_iterations", default=10, min=0, max=30, step=1),
                IO.Boolean.Input("scaled", default=False),
                IO.Boolean.Input("inject_comfy_metadata", default=True),
                IO.Combo.Input("log_level", options=["Basic", "Verbose", "Debug"], default="Basic"),
            ],
            outputs=[IO.Model.Output("model", display_name="model")],
            search_aliases=["HSWQ", "FP8", "Quantizer", "ZIT", "NextDiT", "Calibration"],
            essentials_category="Quantization/ZIT",
        )

    @classmethod
    def execute(
        cls,
        model: IO.Model,
        stats_path: IO.String,
        profile_path: IO.String,
        keep_ratio: IO.Float,
        bins: IO.Int,
        num_candidates: IO.Int,
        refinement_iterations: IO.Int,
        scaled: IO.Boolean,
        inject_comfy_metadata: IO.Boolean,
        log_level: IO.Combo,
    ):
        if not hasattr(torch, "float8_e4m3fn"):
            print("[HSWQ] CRITICAL: torch.float8_e4m3fn is not available in this environment.")
            return (model,)

        resolved_path, session_data = load_session_data(stats_path, log_prefix=zit_spec.QUANTIZATION_LOG_PREFIX)
        if session_data is None:
            return (model,)

        layers_data = session_data["layers"]
        profile_data = zit_spec.load_profile_data(profile_path)
        selection_plan = zit_spec.build_selection_plan(layers_data, keep_ratio, profile_data=profile_data)

        print("------------------------------------------------")
        print("[ZIT-HSWQ] FP8 Quantization Start")
        print(f"  Stats: {resolved_path}")
        if profile_path:
            print(f"  Profile: {profile_path}")
        print(
            f"  Layers: {len(selection_plan['sensitivities']) + len(selection_plan.get('hard_veto_layers', set()))}, "
            f"Keep: {len(selection_plan['keep_names'])}"
        )
        print(f"  Optimizer: mode={zit_spec.OPTIMIZER_MODE}, bins={bins}, cands={num_candidates}, iter={refinement_iterations}, scaled={scaled}")
        for line in selection_plan["extra_logs"]:
            print(f"  {line}")
        print("------------------------------------------------")

        work_model = clear_model_wrapper(model)
        diffusion_model = get_diffusion_model(work_model)
        device = comfy.model_management.get_torch_device()
        optimizer = create_optimizer(
            mode=zit_spec.OPTIMIZER_MODE,
            bins=bins,
            num_candidates=num_candidates,
            refinement_iterations=refinement_iterations,
            device=str(device),
        )
        fp8_quantizer = create_fp8_quantizer(mode=zit_spec.OPTIMIZER_MODE, device=str(device))

        run_stats = quantize_diffusion_model(
            diffusion_model,
            layers_data=layers_data,
            keep_names=selection_plan["keep_names"],
            optimizer=optimizer,
            fp8_quantizer=fp8_quantizer,
            device=device,
            scaled=scaled,
            inject_comfy_metadata=inject_comfy_metadata,
            log_level=log_level,
            log_prefix=zit_spec.QUANTIZATION_LOG_PREFIX,
            amax_adjuster=selection_plan["amax_adjuster"],
        )

        print("------------------------------------------------")
        print("[ZIT-HSWQ] Finished")
        print(f"  Converted FP8 : {run_stats.converted}")
        print(f"  Kept FP16     : {run_stats.kept}")
        print(f"  Skipped (No Stats): {run_stats.skipped_no_stats}")
        print(f"  Skipped already-fp8: {run_stats.skipped_already_fp8}")
        print(f"  Failed        : {run_stats.failed}")
        print("------------------------------------------------")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (work_model,)
