import torch
import torch.nn as nn
import os
import sys
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from .hswq_comfy_api import IO

# ------------------------------------------------------------
# Import HSWQ Optimizer Module
# ------------------------------------------------------------
try:
    # if packaged as a module
    from .weighted_histogram_mse import HSWQWeightedHistogramOptimizer
except Exception:
    # if placed in the same folder
    from weighted_histogram_mse import HSWQWeightedHistogramOptimizer

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

class SDXLHSWQFP8QuantizerLegacyNode(IO.ComfyNode):
    """
    HSWQ FP8 Converter (Legacy / Compat Mode):
    - Implements HSWQ V1.2 Standalone logic.
    - Uses 'weighted_histogram_mse' backend.
    - Enforces scaled=False (Standard FP8 clamping) for maximum compatibility.
    - Does NOT inject extra 'comfy_quant' metadata buffers (Pure weight conversion).
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HSWQFP8ConverterNodeLegacy",
            display_name="HSWQ FP8 Converter (Legacy V1.2 Logic)",
            category="Quantization",
            description="Legacy FP8 conversion using HSWQ V1.2 behavior.",
            inputs=[
                IO.Model.Input("model"),
                IO.String.Input("hswq_stats_path", default="output/hswq_stats/sdxl_calib_session_01.pt"),
                IO.Float.Input("keep_ratio", display_name="FP16 Keep Ratio", default=0.25, min=0.0, max=1.0, step=0.05),
                IO.Combo.Input("log_level", options=["Basic", "Verbose", "Debug"], default="Basic"),
            ],
            outputs=[
                IO.Model.Output("model", display_name="model"),
            ],
            search_aliases=["HSWQ", "Legacy", "FP8", "Converter", "SDXL"],
            essentials_category="Quantization",
        )

    @classmethod
    def execute(
        cls,
        model: IO.Model,
        hswq_stats_path: IO.String,
        keep_ratio: IO.Float,
        log_level: IO.Combo,
    ):
        print(f"### [HSWQ Legacy] Starting FP8 Conversion (Standard/Compat Mode) ###")
        
        # 1. Load Stats
        hswq_stats_path = _resolve_stats_path(hswq_stats_path)
        if not os.path.exists(hswq_stats_path):
            print(f"[HSWQ] Error: Stats file not found at {hswq_stats_path}")
            return (model,)

        try:
            session_data = torch.load(hswq_stats_path, map_location="cpu")
        except Exception as e:
            print(f"[HSWQ] Error loading stats: {e}")
            return (model,)

        layers_data = session_data.get("layers", {})
        if not layers_data:
            print("[HSWQ] Error: No layer data found in stats file.")
            return (model,)

        # 2. Sensitivity Analysis (Output Variance)
        # ----------------------------------------------------------------
        sensitivities = []
        for name, stats in layers_data.items():
            count = int(stats.get("out_count", 0))
            if count > 0:
                mean = stats["output_sum"] / count
                sq_mean = stats["output_sq_sum"] / count
                variance = sq_mean - (mean ** 2)
                if variance < 0: variance = 0.0
                sensitivities.append((name, float(variance)))
        
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        total_layers = len(sensitivities)
        num_keep = int(total_layers * keep_ratio)
        keep_names = set(x[0] for x in sensitivities[:num_keep])
        
        print(f"[HSWQ] Total calibrated layers: {total_layers}")
        print(f"[HSWQ] Keeping Top {num_keep} layers in FP16 (Threshold Var: {sensitivities[num_keep-1][1]:.6f})" if num_keep > 0 else "[HSWQ] Keeping 0 layers.")
        
        # 3. Setup Quantization Environment
        # ----------------------------------------------------------------
        work_model = model.clone()
        if isinstance(work_model, ModelPatcher):
            diffusion_model = work_model.model.diffusion_model
        else:
            diffusion_model = work_model.diffusion_model

        device = comfy.model_management.get_torch_device()
        
        # Use the shared optimizer module (Fixed parameters for Legacy/Standard quality)
        optimizer = HSWQWeightedHistogramOptimizer(
            bins=4096,              # High resolution
            num_candidates=100,      # Adequate for standard mode
            refinement_iterations=3, # Standard refinement
            device=str(device)
        )

        converted_count = 0
        skipped_sensitive_count = 0
        skipped_no_stats_count = 0
        skipped_already_fp8 = 0
        
        print("[HSWQ] Starting quantization loop (Mode: scaled=False / Clip-Only)...")

        # 4. Conversion Loop
        # ----------------------------------------------------------------
        for name, module in diffusion_model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            
            if not hasattr(module, "weight") or module.weight is None:
                continue

            # Check if we should skip
            if name not in layers_data:
                skipped_no_stats_count += 1
                continue
            
            if module.weight.dtype == torch.float8_e4m3fn:
                skipped_already_fp8 += 1
                continue

            if name in keep_names:
                skipped_sensitive_count += 1
                # Optional: Ensure it's clean FP16 (or BF16 if originally so)
                # For Legacy behavior, we mostly leave it alone or cast to FP16
                if module.weight.dtype == torch.bfloat16:
                    module.weight.data = module.weight.data.to(torch.float16)
                if module.bias is not None and module.bias.dtype == torch.bfloat16:
                    module.bias.data = module.bias.data.to(torch.float16)
                continue

            # Perform Quantization
            try:
                stats = layers_data[name]
                in_count = int(stats.get("in_count", 0))
                
                weight = module.weight.data.detach()
                importance = None

                # Extract Importance if available
                if in_count > 0 and "input_imp_sum" in stats:
                    imp_sum = stats["input_imp_sum"]
                    # Validation: Shape check
                    if isinstance(imp_sum, torch.Tensor):
                        # Simple check: does importance size match input channels?
                        # Conv2d: (Out, In, K, K) -> In is dim 1
                        # Linear: (Out, In) -> In is dim 1
                        in_channels = weight.shape[1]
                        if imp_sum.numel() >= in_channels:
                            importance = (imp_sum / in_count).float()
                        else:
                            if log_level == "Debug":
                                print(f"[HSWQ] Shape mismatch {name}: W{weight.shape} vs Imp{imp_sum.shape}. Using ones.")

                # Calculate Optimal AMAX using the shared backend
                # KEY: scaled=False ensures compatibility with standard FP8 loaders (no side-channel scale)
                optimal_amax = optimizer.compute_optimal_amax(weight, importance, scaled=False)
                
                # Apply Quantization (Clip -> Cast)
                w_dev = weight.to(device)
                w_clamped = torch.clamp(w_dev, -optimal_amax, optimal_amax)
                w_fp8 = w_clamped.to(torch.float8_e4m3fn)
                
                # Verify Integrity
                w_check = w_fp8.float()
                if not torch.isfinite(w_check).all():
                    print(f"[HSWQ] CRITICAL: NaNs/Inf detected in {name}. Reverting to FP16.")
                    del w_check, w_dev
                    continue
                del w_check
                
                # Write back
                module.weight.data = w_fp8.to(weight.device)
                
                # Bias to FP16
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.float16)

                converted_count += 1
                
                if log_level == "Debug":
                    print(f" [Quant] {name}: amax={optimal_amax:.4f}")
                
                del w_dev, w_clamped

            except Exception as e:
                print(f"[HSWQ] Failed to quantize {name}: {e}")
                import traceback
                traceback.print_exc()

        print("------------------------------------------------")
        print(f"### [HSWQ Legacy] Finished ###")
        print(f"  Converted FP8 : {converted_count}")
        print(f"  Kept FP16     : {skipped_sensitive_count}")
        print(f"  Skipped (No Stats/FP8): {skipped_no_stats_count + skipped_already_fp8}")
        print("------------------------------------------------")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (work_model,)
