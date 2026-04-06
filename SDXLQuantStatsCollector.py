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
from .model_specs import sdxl as sdxl_spec


def _get_lock(session_key):
    return _get_lock_impl(session_key)


def _atomic_torch_save(obj, path: str):
    _atomic_torch_save_impl(obj, path, log_prefix=sdxl_spec.CALIBRATION_LOG_PREFIX)


def _snapshot_session_for_save(session: dict) -> dict:
    return _snapshot_session_for_save_impl(session)


def _get_session(save_folder_name, file_prefix, session_id):
    return _get_session_impl(
        save_folder_name,
        file_prefix,
        session_id,
        log_prefix=sdxl_spec.CALIBRATION_LOG_PREFIX,
        meta_factory=sdxl_spec.create_session_meta,
    )


class HSWQStatsCollectorBackend(DualMonitorStatsCollectorBackend):
    pass


class SDXLHSWQCalibrationNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SDXLHSWQCalibrationNode",
            display_name="SDXL HSWQ Calibration (DualMonitor V2)",
            category="Quantization",
            description="Collect HSWQ calibration statistics during SDXL sampling.",
            inputs=[
                IO.Model.Input("model"),
                IO.String.Input("save_folder_name", default="hswq_stats"),
                IO.String.Input("file_prefix", default="sdxl_calib"),
                IO.String.Input("session_id", default="session_01"),
                IO.Int.Input("save_every_steps", display_name="Save Every N Steps", default=50, min=1, max=10000),
                IO.Boolean.Input("reset_session", default=False),
            ],
            outputs=[IO.Model.Output("model", display_name="model")],
            search_aliases=["HSWQ", "Calibration", "SDXL", "Stats", "Collector"],
            essentials_category="Quantization",
        )

    @classmethod
    def execute(
        cls,
        model: IO.Model,
        save_folder_name: IO.String,
        file_prefix: IO.String,
        session_id: IO.String,
        save_every_steps: IO.Int,
        reset_session: IO.Boolean,
    ):
        wrapped_model = model.clone()
        device = comfy.model_management.get_torch_device()
        session, ckpt_path, lock = _get_session(save_folder_name, file_prefix, session_id)

        if reset_session:
            _reset_session_impl(session, lock, ckpt_path, session_id, log_prefix=sdxl_spec.CALIBRATION_LOG_PREFIX)

        diffusion_model = wrapped_model.model.diffusion_model
        cleanup_hooks(diffusion_model, sdxl_spec.HOOK_ATTR, log_prefix=sdxl_spec.CALIBRATION_LOG_PREFIX)

        collector_ref = {"collector": None}
        register_hooks(
            diffusion_model,
            hook_attr=sdxl_spec.HOOK_ATTR,
            collector_ref=collector_ref,
            should_hook=sdxl_spec.should_hook_layer,
            log_prefix=sdxl_spec.CALIBRATION_LOG_PREFIX,
            session_id=session_id,
        )

        wrapped_model.set_model_unet_function_wrapper(
            build_stats_wrapper(
                session=session,
                lock=lock,
                device=device,
                save_every_steps=save_every_steps,
                ckpt_path=ckpt_path,
                collector_ref=collector_ref,
                log_prefix=sdxl_spec.CALIBRATION_LOG_PREFIX,
                backend_cls=HSWQStatsCollectorBackend,
            )
        )
        return (wrapped_model,)
