from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path


TEST_OUTPUT_DIR = tempfile.mkdtemp(prefix="hswq-tests-")


def _install_stub_modules() -> None:
    if "folder_paths" not in sys.modules:
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: TEST_OUTPUT_DIR
        sys.modules["folder_paths"] = folder_paths

    if "comfy" not in sys.modules:
        comfy_pkg = types.ModuleType("comfy")
        comfy_pkg.__path__ = []
        sys.modules["comfy"] = comfy_pkg

    if "comfy.model_management" not in sys.modules:
        model_management = types.ModuleType("comfy.model_management")
        model_management.get_torch_device = lambda: "cpu"
        sys.modules["comfy.model_management"] = model_management
        sys.modules["comfy"].model_management = model_management

    if "comfy.model_patcher" not in sys.modules:
        model_patcher = types.ModuleType("comfy.model_patcher")

        class ModelPatcher:
            pass

        model_patcher.ModelPatcher = ModelPatcher
        sys.modules["comfy.model_patcher"] = model_patcher
        sys.modules["comfy"].model_patcher = model_patcher

    def _input_factory(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    class _IO:
        class ComfyNode:
            pass

        class Schema:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Model:
            Input = staticmethod(_input_factory)
            Output = staticmethod(_input_factory)

        class String:
            Input = staticmethod(_input_factory)
            Output = staticmethod(_input_factory)

        class Int:
            Input = staticmethod(_input_factory)
            Output = staticmethod(_input_factory)

        class Boolean:
            Input = staticmethod(_input_factory)
            Output = staticmethod(_input_factory)

        class Float:
            Input = staticmethod(_input_factory)
            Output = staticmethod(_input_factory)

        class Combo:
            Input = staticmethod(_input_factory)
            Output = staticmethod(_input_factory)

        class Image:
            Input = staticmethod(_input_factory)
            Output = staticmethod(_input_factory)

    class _ComfyExtension:
        async def on_load(self):
            return None

        async def get_node_list(self):
            return []

    for name in ("comfy_api.latest", "comfy_api"):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.IO = _IO
        module.ComfyExtension = _ComfyExtension
        sys.modules[name] = module

    if "comfy_api" not in sys.modules:
        comfy_api = types.ModuleType("comfy_api")
        comfy_api.__path__ = []
        sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api"].IO = _IO
    sys.modules["comfy_api"].ComfyExtension = _ComfyExtension
    sys.modules["comfy_api"].latest = sys.modules["comfy_api.latest"]


def load_package_module(module_name: str):
    _install_stub_modules()

    repo_root = Path(__file__).resolve().parents[1]
    package_name = "hswq_pkg"

    module_path = repo_root.joinpath(*module_name.split("."))
    if module_path.is_dir():
        file_path = module_path / "__init__.py"
    else:
        file_path = module_path.with_suffix(".py")

    is_root_package = module_name == "__init__"
    full_name = package_name if is_root_package else f"{package_name}.{module_name}"
    if full_name in sys.modules:
        if is_root_package and getattr(sys.modules[full_name], "__file__", None) is None:
            del sys.modules[full_name]
        else:
            return sys.modules[full_name]

    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(repo_root)]
        sys.modules[package_name] = pkg

    if is_root_package and package_name in sys.modules and getattr(sys.modules[package_name], "__file__", None) is None:
        del sys.modules[package_name]

    if full_name in sys.modules:
        return sys.modules[full_name]

    parent_module = package_name
    parent_path = repo_root
    for part in module_name.split(".")[:-1]:
        parent_module = f"{parent_module}.{part}"
        parent_path = parent_path / part
        if parent_module not in sys.modules:
            pkg = types.ModuleType(parent_module)
            pkg.__path__ = [str(parent_path)]
            sys.modules[parent_module] = pkg

    spec = importlib.util.spec_from_file_location(
        full_name,
        file_path,
        submodule_search_locations=[str(repo_root)] if is_root_package else None,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module
