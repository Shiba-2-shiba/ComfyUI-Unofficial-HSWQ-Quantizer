from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from test_support import TEST_OUTPUT_DIR, load_package_module


class QuantizerHelperTests(unittest.TestCase):
    def test_encode_comfy_quant_round_trips_to_json_bytes(self):
        module = load_package_module("SDXLHSWQQuantizer")
        encoded = module._encode_comfy_quant()
        decoded = bytes(encoded.tolist()).decode("utf-8")
        self.assertEqual(json.loads(decoded), {"format": "float8_e4m3fn"})

    def test_del_buffer_removes_registered_buffer(self):
        module = load_package_module("SDXLHSWQQuantizer")
        layer = nn.Linear(2, 2)
        layer.register_buffer("weight_scale", torch.tensor(1.0))
        module._del_buffer(layer, "weight_scale")
        self.assertNotIn("weight_scale", layer._buffers)

    def test_resolve_stats_path_checks_output_directory_fallback(self):
        module = load_package_module("ZITHSWQQuantizer")
        stats_dir = Path(TEST_OUTPUT_DIR)
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / "fallback_stats.pt"
        stats_path.write_bytes(b"test")

        resolved = module._resolve_stats_path("fallback_stats.pt")

        self.assertEqual(Path(resolved), stats_path)


if __name__ == "__main__":
    unittest.main()
