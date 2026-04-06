import comfy.model_management

from .core.calibration_common import (
    DualMonitorStatsCollectorBackend,
    build_stats_wrapper,
    cleanup_hooks,
    register_hooks,
)
from .core.session_store import (
    _SESSIONS,
    _SESSION_LOCKS,
    atomic_torch_save as _atomic_torch_save_impl,
    get_lock as _get_lock_impl,
    get_session as _get_session_impl,
    reset_session as _reset_session_impl,
    snapshot_session_for_save as _snapshot_session_for_save_impl,
)
from .hswq_comfy_api import IO
from .model_specs import zit as zit_spec


def _get_lock(session_key):
    return _get_lock_impl(session_key)


def _atomic_torch_save(obj, path: str):
    _atomic_torch_save_impl(obj, path, log_prefix=zit_spec.CALIBRATION_LOG_PREFIX)


def _snapshot_session_for_save(session: dict) -> dict:
    return _snapshot_session_for_save_impl(session)


def _get_session(save_folder_name, file_prefix, session_id):
    return _get_session_impl(
        save_folder_name,
        file_prefix,
        session_id,
        log_prefix=zit_spec.CALIBRATION_LOG_PREFIX,
        meta_factory=zit_spec.create_session_meta,
    )


class ZITStatsCollectorBackend(DualMonitorStatsCollectorBackend):
    pass


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
                IO.Combo.Input("target_layer", options=zit_spec.TARGET_LAYER_OPTIONS, default="all_linear_conv"),
                IO.Int.Input("save_every_steps", default=50, min=1, max=10000),
                IO.Boolean.Input("reset_session", default=False),
            ],
            outputs=[IO.Model.Output("model", display_name="model")],
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
        wrapped_model = model.clone()
        device = comfy.model_management.get_torch_device()
        session, ckpt_path, lock = _get_session(save_folder_name, file_prefix, session_id)

        if reset_session:
            _reset_session_impl(session, lock, ckpt_path, session_id, log_prefix=zit_spec.CALIBRATION_LOG_PREFIX)

        diffusion_model = wrapped_model.model.diffusion_model
        cleanup_hooks(diffusion_model, zit_spec.HOOK_ATTR, log_prefix=zit_spec.CALIBRATION_LOG_PREFIX)

        collector_ref = {"collector": None}
        register_hooks(
            diffusion_model,
            hook_attr=zit_spec.HOOK_ATTR,
            collector_ref=collector_ref,
            should_hook=lambda layer_name: zit_spec.should_hook_layer(layer_name, target_layer),
            log_prefix=zit_spec.CALIBRATION_LOG_PREFIX,
            session_id=f"{session_id} (target={target_layer})",
        )

        wrapped_model.set_model_unet_function_wrapper(
            build_stats_wrapper(
                session=session,
                lock=lock,
                device=device,
                save_every_steps=save_every_steps,
                ckpt_path=ckpt_path,
                collector_ref=collector_ref,
                log_prefix=zit_spec.CALIBRATION_LOG_PREFIX,
                backend_cls=ZITStatsCollectorBackend,
                extra_save_paths=lambda current_steps, path: [path.replace(".pt", f"_step{current_steps:06d}.pt")],
            )
        )
        return (wrapped_model,)
