import torch
import torch.nn as nn
import os
import time
import threading
import comfy.model_management
import folder_paths
from .hswq_comfy_api import IO

# ----------------------------------------------------------------------------
# グローバルセッション管理
# ----------------------------------------------------------------------------
_SESSIONS = {}
_SESSION_LOCKS = {}
_GLOBAL_LOCK = threading.Lock()

def _get_lock(session_key):
    with _GLOBAL_LOCK:
        if session_key not in _SESSION_LOCKS:
            _SESSION_LOCKS[session_key] = threading.Lock()
        return _SESSION_LOCKS[session_key]

def _atomic_torch_save(obj, path: str):
    """書き込み中の破損を防ぐAtomic Save (os.replace使用)"""
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        # WindowsでもAtomicに近い動作を期待して replace を使用
        if os.path.exists(path):
            os.replace(tmp, path)
        else:
            os.rename(tmp, path)
    except Exception as e:
        print(f"[ZITCollector] Save failed: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _snapshot_session_for_save(session: dict) -> dict:
    """保存用スナップショットを作成する（保存中に内容が変わらないように固定化）
    注意: dict の shallow copy だけだと、Tensor が参照共有のままになり保存中に in-place 更新され得る
    """
    meta = dict(session.get("meta", {}))
    layers_in = session.get("layers", {})
    layers_out = {}
    
    for name, st in layers_in.items():
        imp = st.get("input_imp_sum", None)
        # Tensorはcloneして計算グラフから切り離し、メモリを別にする
        if isinstance(imp, torch.Tensor):
            imp = imp.detach().clone()
            
        layers_out[name] = {
            "output_sum": float(st.get("output_sum", 0.0)),
            "output_sq_sum": float(st.get("output_sq_sum", 0.0)),
            "out_count": int(st.get("out_count", 0)),
            "input_imp_sum": imp,
            "in_count": int(st.get("in_count", 0)),
        }
    return {"meta": meta, "layers": layers_out}

def _get_session(save_folder_name, file_prefix, session_id):
    """セッションの取得・初期化・ロード"""
    key = f"{save_folder_name}::{file_prefix}::{session_id}"
    lock = _get_lock(key)

    output_dir = folder_paths.get_output_directory()
    full_output_path = os.path.join(output_dir, save_folder_name)
    os.makedirs(full_output_path, exist_ok=True)
    
    ckpt_path = os.path.join(full_output_path, f"{file_prefix}_{session_id}.pt")

    with lock:
        # 1. メモリ上のキャッシュを確認
        if key in _SESSIONS:
            return _SESSIONS[key], ckpt_path, lock

        # 2. ディスクから復元
        if os.path.exists(ckpt_path):
            try:
                print(f"[ZITCollector] Loading session from {ckpt_path}")
                data = torch.load(ckpt_path, map_location="cpu")
                # HSWQ V2 形式チェック
                if data.get("meta", {}).get("type") != "hswq_dual_monitor_v2":
                    print("[ZITCollector] Warning: Legacy/Mismatch file type. Starting new session (V2 High Precision).")
                else:
                    _SESSIONS[key] = data
                    return data, ckpt_path, lock
            except Exception as e:
                print(f"[ZITCollector] Error loading checkpoint: {e}")

        # 3. 新規作成
        print(f"[ZITCollector] Starting new session: {session_id}")
        session_data = {
            "meta": {
                "type": "hswq_dual_monitor_v2", # SDXL版と統一
                "model_type": "NextDiT",        # モデル種別は区別用に残す
                "created_at": time.strftime("%Y%m%d_%H%M%S"),
                "total_steps": 0,
            },
            "layers": {} 
        }
        _SESSIONS[key] = session_data
        return session_data, ckpt_path, lock

# ----------------------------------------------------------------------------
# 集計バックエンド (HSWQ DualMonitor V2 - High Precision)
# ----------------------------------------------------------------------------
class ZITStatsCollectorBackend:
    def __init__(self, session, lock, device):
        self.session = session
        self.lock = lock
        self.device = device

    def hook_fn(self, module, input_t, output_t, name):
        inp = input_t[0] if isinstance(input_t, tuple) else input_t
        out = output_t
        
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            return

        # --- 1. Output Sensitivity Calculation (Variance) ---
        out_f32 = out.detach().float()
        batch_mean = out_f32.mean().item()
        batch_sq_mean = (out_f32 ** 2).mean().item()
        
        # --- 2. Input Importance Calculation (Channel Mean Abs) ---
        inp_detached = inp.detach()
        
        # ZIT/NextDiT (Transformer) vs Conv2d 対応
        if inp_detached.dim() == 4:   # Conv2d (B, C, H, W)
            current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
        elif inp_detached.dim() == 3: # Transformer (B, T, C)
            current_imp = inp_detached.abs().mean(dim=(0, 1))
        elif inp_detached.dim() == 2: # Linear (B, C)
             current_imp = inp_detached.abs().mean(dim=0)
        else:
            current_imp = torch.ones((1,), device=inp_detached.device, dtype=inp_detached.dtype)

        # CPU float64 (Double) で集計して精度を確保 [SDXL版準拠]
        current_imp_cpu = current_imp.to(device="cpu", dtype=torch.float64)
        
        with self.lock:
            layers = self.session["layers"]
            if name not in layers:
                layers[name] = {
                    "output_sum": 0.0,
                    "output_sq_sum": 0.0,
                    "out_count": 0,
                    "input_imp_sum": torch.zeros_like(current_imp_cpu, dtype=torch.float64),
                    "in_count": 0
                }
            
            l_stats = layers[name]
            l_stats["output_sum"] += batch_mean
            l_stats["output_sq_sum"] += batch_sq_mean
            l_stats["out_count"] += 1
            
            # shape check & add
            if l_stats["input_imp_sum"].shape == current_imp_cpu.shape:
                l_stats["input_imp_sum"].add_(current_imp_cpu)
                l_stats["in_count"] += 1

# ----------------------------------------------------------------------------
# ComfyUI ノード定義
# ----------------------------------------------------------------------------
class ZITHSWQCalibrationNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ZITHSWQCalibrationNode",
            display_name="ZIT HSWQ Calibration (DualMonitor V2)",
            category="ZIT/Quantization",
            description="Collect HSWQ calibration stats for ZIT/NextDiT models.",
            inputs=[
                IO.Model.Input("model"),
                IO.String.Input("save_folder_name", default="zit_hswq_stats"),
                IO.String.Input("file_prefix", default="zit_calib"),
                IO.String.Input("session_id", default="session_01"),
                IO.Combo.Input(
                    "target_layer",
                    options=["all_linear_conv", "attention_only", "feed_forward_only", "context_refiner"],
                    default="all_linear_conv",
                ),
                IO.Int.Input("save_every_steps", default=50, min=1, max=10000),
                IO.Boolean.Input("reset_session", default=False),
            ],
            outputs=[
                IO.Model.Output("model", display_name="model"),
            ],
            search_aliases=["HSWQ", "Calibration", "ZIT", "NextDiT", "Stats", "Collector"],
            essentials_category="Quantization/ZIT",
        )

    @classmethod
    def execute(
        cls,
        model: IO.Model,
        save_folder_name: IO.String,
        file_prefix: IO.String,
        session_id: IO.String,
        target_layer: IO.Combo,
        save_every_steps: IO.Int,
        reset_session: IO.Boolean,
    ):
        m = model.clone()
        device = comfy.model_management.get_torch_device()
        
        session, ckpt_path, lock = _get_session(save_folder_name, file_prefix, session_id)
        
        if reset_session:
            with lock:
                print(f"[ZITCollector] Resetting session {session_id}")
                session["layers"] = {}
                session["meta"]["total_steps"] = 0
                if os.path.exists(ckpt_path):
                    try: os.remove(ckpt_path)
                    except: pass

        diffusion_model = m.model.diffusion_model
        
        # ----------------------------------------------------------------
        # 1. フックのクリーンアップ (多重登録防止)
        # ----------------------------------------------------------------
        if hasattr(diffusion_model, "_zit_hswq_calibration_hooks"):
            stale_hooks = diffusion_model._zit_hswq_calibration_hooks
            if len(stale_hooks) > 0:
                print(f"[ZITCollector] Cleaning up {len(stale_hooks)} stale hooks from previous run.")
                for h in stale_hooks:
                    try: h.remove()
                    except: pass
            diffusion_model._zit_hswq_calibration_hooks.clear()
        else:
            diffusion_model._zit_hswq_calibration_hooks = []

        # ----------------------------------------------------------------
        # 2. フックの常駐登録 (wrapperでスイッチング) [SDXL方式]
        # ----------------------------------------------------------------
        collector_ref = {"collector": None}

        def _should_hook(layer_name):
            if target_layer == "all_linear_conv": return True
            if target_layer == "attention_only": return ("attn" in layer_name) or ("qkv" in layer_name)
            if target_layer == "feed_forward_only": return ("feed_forward" in layer_name) or ("ffn" in layer_name)
            if target_layer == "context_refiner": return ("context_refiner" in layer_name)
            return True

        def _shared_hook_factory(layer_name):
            def _hook(module, i, o):
                # collectorがセットされている時だけ実行
                c = collector_ref.get("collector", None)
                if c is None: return
                c.hook_fn(module, i, o, layer_name)
            return _hook

        hooks_count = 0
        for name, module in diffusion_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if _should_hook(name):
                    h = module.register_forward_hook(_shared_hook_factory(name))
                    diffusion_model._zit_hswq_calibration_hooks.append(h)
                    hooks_count += 1
        
        print(f"[ZITCollector] Armed {hooks_count} hooks for session {session_id} (target={target_layer})")

        # ----------------------------------------------------------------
        # 3. 実行ラッパー
        # ----------------------------------------------------------------
        def stats_wrapper(model_function, params):
            collector = ZITStatsCollectorBackend(session, lock, device)
            collector_ref["collector"] = collector # Enable hooks

            try:
                input_x = params.get("input")
                timestep = params.get("timestep")
                c = params.get("c")
                
                out = model_function(input_x, timestep, **c)
                
                do_save = False
                with lock:
                    session["meta"]["total_steps"] += 1
                    current_steps = session["meta"]["total_steps"]
                    if current_steps % save_every_steps == 0:
                        do_save = True
                
                if do_save:
                    with lock:
                        save_data = _snapshot_session_for_save(session)
                    _atomic_torch_save(save_data, ckpt_path); _atomic_torch_save(save_data, ckpt_path.replace(".pt", f"_step{current_steps:06d}.pt"))
                    print(f"[ZITCollector] Saved stats at step {current_steps}")

            finally:
                collector_ref["collector"] = None # Disable hooks
            
            return out

        m.set_model_unet_function_wrapper(stats_wrapper)
        return (m, )
