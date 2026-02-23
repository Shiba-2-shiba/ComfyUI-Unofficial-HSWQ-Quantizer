import os, json
import torch
import torch.nn as nn
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from .hswq_comfy_api import IO

# ------------------------------------------------------------
# Import original HSWQ optimizer (Spec-aligned)
# ------------------------------------------------------------
try:
    # if packaged as a module
    from .weighted_histogram_mse import HSWQWeightedHistogramOptimizer, FP8E4M3Quantizer
except Exception:
    # if placed in the same folder
    try:
        from weighted_histogram_mse import HSWQWeightedHistogramOptimizer, FP8E4M3Quantizer
    except Exception:
        HSWQWeightedHistogramOptimizer = None
        FP8E4M3Quantizer = None

def _resolve_stats_path(path: str) -> str:
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

def _encode_comfy_quant(fmt: str = "float8_e4m3fn") -> torch.Tensor:
    b = json.dumps({"format": fmt}).encode("utf-8")
    return torch.tensor(list(b), dtype=torch.uint8)

def _del_buffer(module: nn.Module, name: str):
    if hasattr(module, "_buffers") and name in module._buffers:
        del module._buffers[name]


class ZITHSWQQuantizerNode(IO.ComfyNode):
    """
    ZIT HSWQ FP8 Quantizer (Spec-aligned):
      - sensitivity: output variance ranking (keep_ratio)
      - amax: weighted_histogram_mse.HSWQWeightedHistogramOptimizer
      - quant: clamp -> cast float8 (scaled=False default)
      - metadata: standard ComfyUI format
    """

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
                IO.Float.Input("keep_ratio", default=0.25, min=0.0, max=1.0, step=0.05),
                IO.Int.Input("bins", default=8192, min=512, max=65536, step=512),
                IO.Int.Input("num_candidates", default=1000, min=50, max=5000, step=50),
                IO.Int.Input("refinement_iterations", default=10, min=0, max=30, step=1),
                IO.Boolean.Input("scaled", default=False),
                IO.Boolean.Input("inject_comfy_metadata", default=True),
                IO.Combo.Input("log_level", options=["Basic", "Verbose", "Debug"], default="Basic"),
            ],
            outputs=[
                IO.Model.Output("model", display_name="model"),
            ],
            search_aliases=["HSWQ", "FP8", "Quantizer", "ZIT", "NextDiT", "Calibration"],
            essentials_category="Quantization/ZIT",
        )

    @classmethod
    def execute(
        cls,
        model: IO.Model,
        stats_path: IO.String,
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

        if HSWQWeightedHistogramOptimizer is None:
            print("[HSWQ] CRITICAL: 'weighted_histogram_mse.py' not found. Please place it in the same folder.")
            return (model,)

        stats_path = _resolve_stats_path(stats_path)
        if not os.path.exists(stats_path):
            print(f"[HSWQ] Error: Stats file not found: {stats_path}")
            return (model,)

        try:
            session_data = torch.load(stats_path, map_location="cpu")
        except Exception as e:
            print(f"[HSWQ] Error loading stats: {e}")
            return (model,)

        # Meta Check
        meta = session_data.get("meta", {})
        if meta.get("type") != "hswq_dual_monitor_v2":
             print(f"[HSWQ] Warning: meta.type is '{meta.get('type')}', expected 'hswq_dual_monitor_v2'.")

        layers_data = session_data.get("layers", {})
        if not layers_data:
            print("[HSWQ] Error: No layers found in stats.")
            return (model,)

        # ------------------------------------------------------------
        # 1) Sensitivity Ranking (Variance)
        # ------------------------------------------------------------
        sensitivities = []
        for name, st in layers_data.items():
            c = int(st.get("out_count", 0))
            if c <= 0: continue
            mean = st["output_sum"] / c
            sq_mean = st["output_sq_sum"] / c
            var = sq_mean - (mean ** 2)
            if var < 0: var = 0.0
            sensitivities.append((name, float(var)))

        sensitivities.sort(key=lambda x: x[1], reverse=True)
        total = len(sensitivities)
        num_keep = int(total * float(keep_ratio))
        keep_names = set(n for n, _ in sensitivities[:num_keep])

        print("------------------------------------------------")
        print("[ZIT-HSWQ] FP8 Quantization Start")
        print(f"  Stats: {stats_path}")
        print(f"  Layers: {total}, Keep: {num_keep} (Top {keep_ratio*100:.1f}%)")
        print(f"  Optimizer: bins={bins}, cands={num_candidates}, iter={refinement_iterations}, scaled={scaled}")
        print("------------------------------------------------")

        # ------------------------------------------------------------
        # 2) Prepare Model + Optimizer
        # ------------------------------------------------------------
        work_model = model.clone()

        # Remove calibration wrapper if exists
        if hasattr(work_model, "set_model_unet_function_wrapper"):
            try: work_model.set_model_unet_function_wrapper(None)
            except: pass

        if isinstance(work_model, ModelPatcher):
            diffusion_model = work_model.model.diffusion_model
        else:
            diffusion_model = work_model.diffusion_model

        device = comfy.model_management.get_torch_device()

        optimizer = HSWQWeightedHistogramOptimizer(
            bins=bins,
            num_candidates=num_candidates,
            refinement_iterations=refinement_iterations,
            device=str(device),
        )

        fp8q = FP8E4M3Quantizer(str(device))
        fp8_max = float(getattr(fp8q, "max_representable", 448.0))

        meta_proto = _encode_comfy_quant("float8_e4m3fn")

        converted = 0
        kept = 0
        skipped_no_stats = 0
        skipped_already_fp8 = 0
        failed = 0

        # ------------------------------------------------------------
        # 3) Quantize Loop
        # ------------------------------------------------------------
        for name, module in diffusion_model.named_modules():
            # Target both Linear (Transformer) and Conv2d (PatchEmbed/Output)
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue

            if name not in layers_data:
                skipped_no_stats += 1
                continue

            # Check if already FP8
            if module.weight.dtype == torch.float8_e4m3fn:
                skipped_already_fp8 += 1
                continue

            # Protected Layers (Sensitivity Keep)
            if name in keep_names:
                kept += 1
                _del_buffer(module, "comfy_quant")
                _del_buffer(module, "weight_scale")
                # Normalize to FP16
                if module.weight.dtype == torch.bfloat16:
                    module.weight.data = module.weight.data.to(torch.float16)
                if module.bias is not None and module.bias.dtype == torch.bfloat16:
                    module.bias.data = module.bias.data.to(torch.float16)
                continue

            # Quantize
            st = layers_data[name]
            in_count = int(st.get("in_count", 0))
            
            importance = None
            if in_count > 0 and isinstance(st.get("input_imp_sum", None), torch.Tensor):
                # Collector has float64 sum, normalize to mean
                importance = (st["input_imp_sum"] / in_count).float()

            try:
                w = module.weight.data.detach()
                
                # Compute Optimal Amax via External Optimizer
                amax = float(optimizer.compute_optimal_amax(w, importance, scaled=scaled))
                if not (amax > 0):
                    failed += 1
                    continue

                w_dev = w.to(device=device, dtype=torch.float16)

                if scaled:
                    # High Performance Mode (Use full FP8 range)
                    scale = fp8_max / max(amax, 1e-12)
                    w_scaled = (w_dev * scale).clamp(-fp8_max, fp8_max)
                    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
                    weight_scale = (amax / fp8_max)
                else:
                    # Compatible Mode (Clamp only)
                    clip = min(amax, fp8_max)
                    w_clamped = w_dev.clamp(-clip, clip)
                    w_fp8 = w_clamped.to(torch.float8_e4m3fn)
                    weight_scale = 1.0

                # Verify Finite
                if not torch.isfinite(w_fp8.float()).all():
                    if log_level in ["Verbose", "Debug"]:
                        print(f"[HSWQ] Reject (non-finite) -> keep FP16: {name}")
                    failed += 1
                    del w_dev
                    continue

                # Apply to Model
                module.weight.data = w_fp8.to(w.device)
                
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.float16)

                # Inject Metadata
                if inject_comfy_metadata:
                    _del_buffer(module, "comfy_quant")
                    _del_buffer(module, "weight_scale")
                    module.register_buffer("comfy_quant", meta_proto.clone().to(w.device))
                    module.register_buffer("weight_scale", torch.tensor(float(weight_scale), dtype=torch.float32, device=w.device))
                else:
                    _del_buffer(module, "comfy_quant")
                    _del_buffer(module, "weight_scale")

                converted += 1
                if log_level == "Debug":
                     print(f"[Quant] {name}: amax={amax:.4f}, scale={weight_scale:.4f}")

                del w_dev

            except Exception as e:
                failed += 1
                if log_level in ["Verbose", "Debug"]:
                    print(f"[HSWQ] Failed: {name} -> {e}")

        print("------------------------------------------------")
        print("[ZIT-HSWQ] Finished")
        print(f"  Converted FP8 : {converted}")
        print(f"  Kept FP16     : {kept}")
        print(f"  Skipped (No Stats): {skipped_no_stats}")
        print(f"  Failed        : {failed}")
        print("------------------------------------------------")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (work_model,)
