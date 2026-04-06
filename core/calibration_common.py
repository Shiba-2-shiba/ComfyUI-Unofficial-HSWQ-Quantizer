from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn

from .session_store import atomic_torch_save, snapshot_session_for_save


def compute_input_importance(input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.dim() == 4:
        return input_tensor.abs().mean(dim=(0, 2, 3))
    if input_tensor.dim() == 3:
        return input_tensor.abs().mean(dim=(0, 1))
    if input_tensor.dim() == 2:
        return input_tensor.abs().mean(dim=0)
    return torch.ones((1,), device=input_tensor.device, dtype=input_tensor.dtype)


class DualMonitorStatsCollectorBackend:
    def __init__(self, session: dict, lock, device):
        self.session = session
        self.lock = lock
        self.device = device

    def hook_fn(self, module, input_t, output_t, name: str) -> None:
        input_tensor = input_t[0] if isinstance(input_t, tuple) else input_t
        output_tensor = output_t

        if not isinstance(input_tensor, torch.Tensor) or not isinstance(output_tensor, torch.Tensor):
            return

        output_f32 = output_tensor.detach().float()
        batch_mean = output_f32.mean().item()
        batch_sq_mean = (output_f32 ** 2).mean().item()

        current_importance = compute_input_importance(input_tensor.detach()).to(device="cpu", dtype=torch.float64)

        with self.lock:
            layers = self.session["layers"]
            if name not in layers:
                layers[name] = {
                    "output_sum": 0.0,
                    "output_sq_sum": 0.0,
                    "out_count": 0,
                    "input_imp_sum": torch.zeros_like(current_importance, dtype=torch.float64),
                    "in_count": 0,
                }

            layer_stats = layers[name]
            layer_stats["output_sum"] += batch_mean
            layer_stats["output_sq_sum"] += batch_sq_mean
            layer_stats["out_count"] += 1

            if layer_stats["input_imp_sum"].shape == current_importance.shape:
                layer_stats["input_imp_sum"].add_(current_importance)
                layer_stats["in_count"] += 1


def cleanup_hooks(diffusion_model, hook_attr: str, *, log_prefix: str) -> None:
    if hasattr(diffusion_model, hook_attr):
        stale_hooks = getattr(diffusion_model, hook_attr)
        if len(stale_hooks) > 0:
            print(f"[{log_prefix}] Cleaning up {len(stale_hooks)} stale hooks from previous run.")
            for hook in stale_hooks:
                try:
                    hook.remove()
                except Exception:
                    pass
        stale_hooks.clear()
    else:
        setattr(diffusion_model, hook_attr, [])


def register_hooks(
    diffusion_model,
    *,
    hook_attr: str,
    collector_ref: dict,
    should_hook: Callable[[str], bool] | None,
    log_prefix: str,
    session_id: str,
) -> int:
    def shared_hook_factory(layer_name: str):
        def _hook(module, inputs, output):
            collector = collector_ref.get("collector")
            if collector is None:
                return
            collector.hook_fn(module, inputs, output, layer_name)

        return _hook

    hooks_count = 0
    hooks = getattr(diffusion_model, hook_attr)
    for name, module in diffusion_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and (should_hook is None or should_hook(name)):
            hooks.append(module.register_forward_hook(shared_hook_factory(name)))
            hooks_count += 1

    print(f"[{log_prefix}] Armed {hooks_count} hooks for session {session_id}")
    return hooks_count


def build_stats_wrapper(
    *,
    session: dict,
    lock,
    device,
    save_every_steps: int,
    ckpt_path: str,
    collector_ref: dict,
    log_prefix: str,
    backend_cls=DualMonitorStatsCollectorBackend,
    extra_save_paths: Callable[[int, str], Iterable[str]] | None = None,
):
    def stats_wrapper(model_function, params):
        collector = backend_cls(session, lock, device)
        collector_ref["collector"] = collector

        try:
            input_x = params.get("input")
            timestep = params.get("timestep")
            conditioning = params.get("c")

            output = model_function(input_x, timestep, **conditioning)

            do_save = False
            with lock:
                session["meta"]["total_steps"] += 1
                current_steps = session["meta"]["total_steps"]
                if current_steps % save_every_steps == 0:
                    do_save = True

            if do_save:
                with lock:
                    save_data = snapshot_session_for_save(session)
                atomic_torch_save(save_data, ckpt_path, log_prefix=log_prefix)
                for extra_path in extra_save_paths(current_steps, ckpt_path) if extra_save_paths else ():
                    atomic_torch_save(save_data, extra_path, log_prefix=log_prefix)
                print(f"[{log_prefix}] Saved stats at step {current_steps}")
        finally:
            collector_ref["collector"] = None

        return output

    return stats_wrapper

