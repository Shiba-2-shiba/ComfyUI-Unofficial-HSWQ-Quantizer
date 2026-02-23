from __future__ import annotations

import os
import traceback

_LOG_NAME = "hswq_import_error.log"


def _write_import_error(message: str) -> None:
    try:
        path = os.path.join(os.path.dirname(__file__), _LOG_NAME)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message)
            if not message.endswith("\n"):
                handle.write("\n")
    except Exception:
        pass


try:
    from comfy_api.latest import IO, ComfyExtension  # type: ignore
    SOURCE = "comfy_api.latest"
except Exception:
    try:
        from comfy_api import IO, ComfyExtension  # type: ignore
        SOURCE = "comfy_api"
        _write_import_error(
            "[HSWQ] Fallback to comfy_api (comfy_api.latest not available)."
        )
    except Exception:
        _write_import_error(
            "[HSWQ] Failed to import comfy_api.latest and comfy_api.\n"
            + traceback.format_exc()
        )
        raise


__all__ = ["IO", "ComfyExtension", "SOURCE", "_write_import_error"]
