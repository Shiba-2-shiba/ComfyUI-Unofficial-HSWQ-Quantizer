from __future__ import annotations

import asyncio
import unittest

import torch

from test_support import load_package_module


class RefactorCoreTests(unittest.TestCase):
    def test_optimizer_bridge_supports_classic_and_fast_modes(self):
        module = load_package_module("core.optimizer_bridge")
        weight = torch.tensor([[0.25, -0.75], [1.25, -1.75]], dtype=torch.float32)
        importance = torch.tensor([1.0, 2.0], dtype=torch.float32)

        classic = module.create_optimizer(mode="classic", bins=64, num_candidates=10, refinement_iterations=1, device="cpu")
        fast = module.create_optimizer(mode="fast", bins=64, num_candidates=10, refinement_iterations=1, device="cpu")

        classic_amax = classic.compute_optimal_amax(weight, importance, scaled=False)
        fast_amax = fast.compute_optimal_amax(weight, importance, scaled=False)

        self.assertGreater(classic_amax, 0.0)
        self.assertGreater(fast_amax, 0.0)

    def test_zit_selection_plan_preserves_hard_veto_and_dynamic_floor(self):
        module = load_package_module("model_specs.zit")
        layers_data = {
            "layer.safe": {"output_sum": 1.0, "output_sq_sum": 2.0, "out_count": 1, "input_imp_sum": torch.ones(2), "in_count": 1},
            "layer.veto": {"output_sum": 0.1, "output_sq_sum": 0.2, "out_count": 1, "input_imp_sum": torch.ones(2), "in_count": 1},
        }
        profile_data = {
            "layer.veto.weight": {"kurtosis": 25.0, "outlier_ratio": 10.0, "abs_max": 5.0},
            "layer.safe.weight": {"kurtosis": 1.0, "outlier_ratio": 2.0, "abs_max": 1.0},
        }

        plan = module.build_selection_plan(layers_data, keep_ratio=0.5, profile_data=profile_data)

        self.assertIn("layer.veto", plan["keep_names"])
        adjusted = plan["amax_adjuster"]("layer.safe", torch.tensor([2.0, -4.0]), 0.25)
        self.assertGreaterEqual(adjusted, 2.0)

    def test_package_entrypoint_exposes_all_public_nodes(self):
        module = load_package_module("__init__")
        ext = asyncio.run(module.comfy_entrypoint())
        node_names = {node.__name__ for node in asyncio.run(ext.get_node_list())}

        self.assertIn("SDXLHSWQCalibrationNode", node_names)
        self.assertIn("SDXLHSWQFP8QuantizerNode", node_names)
        self.assertIn("SDXLHSWQFP8QuantizerLegacyNode", node_names)
        self.assertIn("ZITHSWQCalibrationNode", node_names)
        self.assertIn("ZITHSWQQuantizerNode", node_names)
        self.assertIn("HSWQAdvancedBenchmark", node_names)


if __name__ == "__main__":
    unittest.main()
