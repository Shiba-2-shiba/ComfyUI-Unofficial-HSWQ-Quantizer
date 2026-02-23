import traceback

from .hswq_comfy_api import ComfyExtension, _write_import_error

_NODE_IMPORTS = [
    ("SDXLQuantStatsCollector", "SDXLHSWQCalibrationNode"),
    ("SDXLHSWQQuantizer", "SDXLHSWQFP8QuantizerNode"),
    ("SDXLHSWQQuantizerLegacy", "SDXLHSWQFP8QuantizerLegacyNode"),
    ("ZITQuantStatsCollector", "ZITHSWQCalibrationNode"),
    ("ZITHSWQQuantizer", "ZITHSWQQuantizerNode"),
    ("HSWQAdvancedBenchmark", "HSWQAdvancedBenchmark"),
]

_NODE_LIST = []
_IMPORT_ERRORS = []

for module_name, class_name in _NODE_IMPORTS:
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
        _NODE_LIST.append(getattr(module, class_name))
    except Exception:
        _IMPORT_ERRORS.append((module_name, class_name, traceback.format_exc()))

if _IMPORT_ERRORS:
    lines = ["[HSWQ] Node import errors:"]
    for module_name, class_name, err in _IMPORT_ERRORS:
        lines.append(f"- {module_name}.{class_name} failed:")
        lines.append(err)
    _write_import_error("\n".join(lines))


class HSWQExtension(ComfyExtension):
    async def on_load(self) -> None:
        print("### HSWQ Nodes: Initializing... ###")
        if _NODE_LIST:
            for node in _NODE_LIST:
                print(f"  [OK] {node.__name__} loaded.")
        if _IMPORT_ERRORS:
            print("  [ERROR] Some HSWQ nodes failed to import.")
            print("  See hswq_import_error.log for details.")
        print(f"### HSWQ Nodes: Initialization complete. Total nodes: {len(_NODE_LIST)} ###")

    async def get_node_list(self):
        return list(_NODE_LIST)


async def comfy_entrypoint():
    return HSWQExtension()


__all__ = ["comfy_entrypoint"]
