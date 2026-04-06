from __future__ import annotations

import os
import threading
import time
from typing import Callable

import folder_paths
import torch


SESSION_FILE_TYPE = "hswq_dual_monitor_v2"

_SESSIONS: dict[str, dict] = {}
_SESSION_LOCKS: dict[str, threading.Lock] = {}
_GLOBAL_LOCK = threading.Lock()


def get_lock(session_key: str) -> threading.Lock:
    with _GLOBAL_LOCK:
        if session_key not in _SESSION_LOCKS:
            _SESSION_LOCKS[session_key] = threading.Lock()
        return _SESSION_LOCKS[session_key]


def atomic_torch_save(obj, path: str, *, log_prefix: str) -> None:
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        if os.path.exists(path):
            os.replace(tmp, path)
        else:
            os.rename(tmp, path)
    except Exception as exc:
        print(f"[{log_prefix}] Save failed: {exc}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def snapshot_session_for_save(session: dict) -> dict:
    meta = dict(session.get("meta", {}))
    layers_out = {}

    for name, stats in session.get("layers", {}).items():
        input_imp_sum = stats.get("input_imp_sum")
        if isinstance(input_imp_sum, torch.Tensor):
            input_imp_sum = input_imp_sum.detach().clone()

        layers_out[name] = {
            "output_sum": float(stats.get("output_sum", 0.0)),
            "output_sq_sum": float(stats.get("output_sq_sum", 0.0)),
            "out_count": int(stats.get("out_count", 0)),
            "input_imp_sum": input_imp_sum,
            "in_count": int(stats.get("in_count", 0)),
        }

    return {"meta": meta, "layers": layers_out}


def build_default_meta(**extra_meta) -> dict:
    meta = {
        "type": SESSION_FILE_TYPE,
        "created_at": time.strftime("%Y%m%d_%H%M%S"),
        "total_steps": 0,
    }
    meta.update(extra_meta)
    return meta


def get_session(
    save_folder_name: str,
    file_prefix: str,
    session_id: str,
    *,
    log_prefix: str,
    meta_factory: Callable[[], dict] | None = None,
):
    key = f"{save_folder_name}::{file_prefix}::{session_id}"
    lock = get_lock(key)

    output_dir = folder_paths.get_output_directory()
    full_output_path = os.path.join(output_dir, save_folder_name)
    os.makedirs(full_output_path, exist_ok=True)

    ckpt_path = os.path.join(full_output_path, f"{file_prefix}_{session_id}.pt")

    with lock:
        if key in _SESSIONS:
            return _SESSIONS[key], ckpt_path, lock

        if os.path.exists(ckpt_path):
            try:
                print(f"[{log_prefix}] Loading session from {ckpt_path}")
                data = torch.load(ckpt_path, map_location="cpu")
                if data.get("meta", {}).get("type") != SESSION_FILE_TYPE:
                    print(f"[{log_prefix}] Warning: Legacy/Mismatch file type. Starting new session.")
                else:
                    _SESSIONS[key] = data
                    return data, ckpt_path, lock
            except Exception as exc:
                print(f"[{log_prefix}] Error loading checkpoint: {exc}")

        print(f"[{log_prefix}] Starting new session: {session_id}")
        session_data = {
            "meta": meta_factory() if meta_factory is not None else build_default_meta(),
            "layers": {},
        }
        _SESSIONS[key] = session_data
        return session_data, ckpt_path, lock


def reset_session(session: dict, lock: threading.Lock, ckpt_path: str, session_id: str, *, log_prefix: str) -> None:
    with lock:
        print(f"[{log_prefix}] Resetting session {session_id}")
        session["layers"] = {}
        session.setdefault("meta", {})["total_steps"] = 0
        if os.path.exists(ckpt_path):
            try:
                os.remove(ckpt_path)
            except Exception:
                pass

