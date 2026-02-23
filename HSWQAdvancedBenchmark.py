import torch
import torch.nn.functional as F
import numpy as np
import comfy.model_management
from .hswq_comfy_api import IO

# --- Dependency Check ---
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# --- Helper: PyTorch SSIM Implementation (GPU Native) ---
def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)

def ssim_tensor(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = gaussian_window(window_size, 1.5).to(img1.device).type_as(img1)
    window = window.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class HSWQAdvancedBenchmark(IO.ComfyNode):
    _lpips_model = None
    _clip_model = None
    _device = None

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HSWQAdvancedBenchmark",
            display_name="HSWQ Quality Benchmark (Auto-Align)",
            category="Quantization/Benchmark",
            description="Benchmark quality between reference and target images with LPIPS/SSIM/CLIP metrics.",
            inputs=[
                IO.Image.Input("image_ref"),
                IO.Image.Input("image_target"),
                IO.Boolean.Input("compute_lpips", display_name="Compute LPIPS (Texture)", default=True),
                IO.Boolean.Input("compute_clip", display_name="Compute CLIP Score (Semantic)", default=True),
                IO.Boolean.Input("enable_auto_align", display_name="Auto-Align Images (Fix Shift)", default=True),
                IO.Float.Input("diff_amplification", default=1.0, min=1.0, max=100.0),
            ],
            outputs=[
                IO.Image.Output("diff_image", display_name="diff_image"),
                IO.String.Output("report_text", display_name="report_text"),
                IO.Float.Output("lpips_dist", display_name="lpips_dist"),
                IO.Float.Output("ssim_score", display_name="ssim_score"),
                IO.Float.Output("clip_similarity", display_name="clip_similarity"),
                IO.Float.Output("mse_error", display_name="mse_error"),
            ],
            search_aliases=["HSWQ", "Benchmark", "LPIPS", "SSIM", "CLIP", "Diff"],
            essentials_category="Quantization/Benchmark",
        )

    @classmethod
    def _get_device(cls):
        if cls._device is None:
            cls._device = comfy.model_management.get_torch_device()
        return cls._device

    @classmethod
    def load_lpips(cls):
        if cls._lpips_model is None and LPIPS_AVAILABLE:
            print("[HSWQ Bench] Loading LPIPS model (AlexNet)...")
            cls._lpips_model = lpips.LPIPS(net='alex', verbose=False).to(cls._get_device())
            cls._lpips_model.eval()

    @classmethod
    def load_clip(cls):
        if cls._clip_model is None and CLIP_AVAILABLE:
            print("[HSWQ Bench] Loading CLIP model (ViT-B-32)...")
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            cls._clip_model = model.to(cls._get_device())
            cls._clip_model.eval()

    @staticmethod
    def align_images(img_ref, img_target, max_shift=16):
        """
        Brute-force alignment to find best shift (dx, dy) that minimizes MSE.
        img_ref, img_target: (1, H, W, C)
        Returns: aligned_target (cropped), cropped_ref, (dx, dy)
        """
        # Convert to NCHW for easier slicing
        t1 = img_ref.permute(0, 3, 1, 2)
        t2 = img_target.permute(0, 3, 1, 2)
        
        best_mse = float('inf')
        best_shift = (0, 0)
        H, W = t1.shape[2], t1.shape[3]

        # Use center crop for faster alignment check to avoid border artifacts
        check_H = H // 2
        check_W = W // 2
        
        # Simple grid search
        # This runs on GPU so it's reasonably fast for small shifts
        base_t1 = t1[:, :, H//4 : H//4+check_H, W//4 : W//4+check_W]

        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                # Slice t2
                y_start = H//4 + dy
                x_start = W//4 + dx
                
                # Boundary check
                if y_start < 0 or y_start + check_H > H or x_start < 0 or x_start + check_W > W:
                    continue

                curr_t2 = t2[:, :, y_start : y_start+check_H, x_start : x_start+check_W]
                mse = torch.mean((base_t1 - curr_t2) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_shift = (dx, dy)
        
        dx, dy = best_shift
        
        # Now Apply best shift and Crop both to valid intersection
        # To keep dimensions same, we crop the intersection
        
        # Ref bounds
        r_y1 = max(0, -dy)
        r_y2 = min(H, H - dy)
        r_x1 = max(0, -dx)
        r_x2 = min(W, W - dx)
        
        # Target bounds
        t_y1 = max(0, dy)
        t_y2 = min(H, H + dy)
        t_x1 = max(0, dx)
        t_x2 = min(W, W + dx)
        
        # Determine intersection size
        out_h = min(r_y2 - r_y1, t_y2 - t_y1)
        out_w = min(r_x2 - r_x1, t_x2 - t_x1)
        
        ref_cropped = t1[:, :, r_y1:r_y1+out_h, r_x1:r_x1+out_w]
        tgt_cropped = t2[:, :, t_y1:t_y1+out_h, t_x1:t_x1+out_w]
        
        # Return to BHWC
        return tgt_cropped.permute(0, 2, 3, 1), ref_cropped.permute(0, 2, 3, 1), best_shift

    @classmethod
    def execute(
        cls,
        image_ref: IO.Image,
        image_target: IO.Image,
        compute_lpips: IO.Boolean,
        compute_clip: IO.Boolean,
        enable_auto_align: IO.Boolean,
        diff_amplification: IO.Float,
    ):
        device = cls._get_device()
        # 1. Validation & Resize
        if image_ref.shape != image_target.shape:
            if image_ref.shape[1:] != image_target.shape[1:]:
                # Resize target to match ref
                t_p = image_target.permute(0, 3, 1, 2)
                r_H, r_W = image_ref.shape[1], image_ref.shape[2]
                t_resized = F.interpolate(t_p, size=(r_H, r_W), mode='bilinear', align_corners=False)
                image_target = t_resized.permute(0, 2, 3, 1)

        batch_size = image_ref.shape[0]
        
        lpips_scores, ssim_scores, clip_scores, mse_scores = [], [], [], []
        diff_images_batch = []
        shifts_log = []

        if compute_lpips:
            cls.load_lpips()
        if compute_clip:
            cls.load_clip()

        for i in range(batch_size):
            img1 = image_ref[i].unsqueeze(0).to(device)   # Ref
            img2 = image_target[i].unsqueeze(0).to(device) # Target

            # --- AUTO ALIGNMENT ---
            shift_info = "0,0"
            if enable_auto_align:
                img2, img1, shift = cls.align_images(img1, img2, max_shift=12)
                shifts_log.append(shift)
                shift_info = f"{shift[0]},{shift[1]}"
                if shift != (0, 0):
                    # print(f"[HSWQ] Corrected shift for image {i}: dx={shift[0]}, dy={shift[1]}")
                    pass
            
            # 1. MSE
            diff = (img1 - img2)
            mse = torch.mean(diff ** 2).item()
            mse_scores.append(mse)

            # Visual Diff (Amplified)
            diff_vis = torch.abs(diff) * diff_amplification
            diff_vis = torch.clamp(diff_vis, 0.0, 1.0)
            # If cropped, pad back to original size for display? Or just return cropped.
            # Returning cropped is safer to visualize alignment success.
            diff_images_batch.append(diff_vis.cpu())

            # Convert to NCHW
            img1_nchw = img1.permute(0, 3, 1, 2)
            img2_nchw = img2.permute(0, 3, 1, 2)

            # 2. SSIM
            ssim_val = ssim_tensor(img1_nchw, img2_nchw).item()
            ssim_scores.append(ssim_val)

            # 3. LPIPS
            if compute_lpips and LPIPS_AVAILABLE and cls._lpips_model:
                t1_lpips = img1_nchw * 2.0 - 1.0
                t2_lpips = img2_nchw * 2.0 - 1.0
                # LPIPS needs standard size often, but AlexNet handles varying sizes. 
                # Ideally resize to avoid border effects if very small crop.
                if t1_lpips.shape[2] < 32 or t1_lpips.shape[3] < 32:
                     # Skip if too small after crop
                     lpips_scores.append(0.0)
                else:
                    with torch.no_grad():
                        l_dist = cls._lpips_model(t1_lpips, t2_lpips).item()
                    lpips_scores.append(l_dist)
            else:
                lpips_scores.append(0.0)

            # 4. CLIP
            if compute_clip and CLIP_AVAILABLE and cls._clip_model:
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

                t1_clip = F.interpolate(img1_nchw, size=(224, 224), mode='bicubic', align_corners=False)
                t2_clip = F.interpolate(img2_nchw, size=(224, 224), mode='bicubic', align_corners=False)

                t1_clip = (t1_clip - mean) / std
                t2_clip = (t2_clip - mean) / std

                with torch.no_grad():
                    feat1 = cls._clip_model.encode_image(t1_clip)
                    feat2 = cls._clip_model.encode_image(t2_clip)
                    feat1 /= feat1.norm(dim=-1, keepdim=True)
                    feat2 /= feat2.norm(dim=-1, keepdim=True)
                    similarity = (feat1 @ feat2.T).item()
                clip_scores.append(similarity)
            else:
                clip_scores.append(1.0)

        # --- Aggregation ---
        avg_mse = sum(mse_scores) / len(mse_scores)
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        avg_lpips = sum(lpips_scores) / len(lpips_scores)
        avg_clip = sum(clip_scores) / len(clip_scores)

        grade = "B"
        if avg_lpips <= 0.03 and avg_ssim >= 0.98: grade = "S (Perfect)"
        elif avg_lpips <= 0.06 and avg_ssim >= 0.95: grade = "A (Excellent)"
        elif avg_lpips <= 0.10: grade = "B (Good)"
        elif avg_lpips <= 0.15: grade = "C (Acceptable)"
        else: grade = "D (Degraded)"
        
        # Check alignment status
        shifted_msg = ""
        if shifts_log:
            # Check if any shift occurred
            is_shifted = any(s != (0,0) for s in shifts_log)
            if is_shifted:
                shifted_msg = f"  [!] Auto-Align Active. Max shift detected: {max(shifts_log, key=lambda x: max(abs(x[0]), abs(x[1])))}"
            else:
                shifted_msg = "  (No shift detected)"

        report =  f"--- HSWQ Benchmark Report ---\n"
        report += f"Samples         : {batch_size}\n"
        report += f"Grade           : {grade}\n"
        report += f"Alignment       : {shifted_msg}\n"
        report += f"-----------------------------\n"
        report += f"LPIPS (Texture) : {avg_lpips:.5f} (Lower=Better)\n"
        report += f"SSIM  (Struct)  : {avg_ssim:.5f} (Higher=Better)\n"
        report += f"CLIP  (Semantic): {avg_clip:.5f} (Higher=Better)\n"
        report += f"MSE   (Pixel)   : {avg_mse:.6f}\n"
        report += f"-----------------------------\n"
        
        # Pad visuals if batch sizes differ due to crop? 
        # For visualization simply finding max size and padding is enough, 
        # but here we return a list of tensors which comfy converts to batch if sizes match.
        # Since we crop all to intersection, sizes might differ per image if shifts differ.
        # To be safe for ComfyUI batch tensor, let's force resize the diffs to input size.
        
        final_diffs = []
        target_H, target_W = image_ref.shape[1], image_ref.shape[2]
        for d in diff_images_batch:
            # d is (1, H', W', C)
            d_p = d.permute(0, 3, 1, 2)
            d_r = F.interpolate(d_p, size=(target_H, target_W), mode='nearest')
            final_diffs.append(d_r.permute(0, 2, 3, 1))
            
        out_diff = torch.cat(final_diffs, dim=0)

        print(report)
        return (out_diff, report, avg_lpips, avg_ssim, avg_clip, avg_mse)
