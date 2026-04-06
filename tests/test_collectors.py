from __future__ import annotations

import threading
import unittest

import torch

from test_support import load_package_module


class CollectorBehaviorTests(unittest.TestCase):
    def test_sdxl_backend_tracks_input_importance_for_multiple_tensor_shapes(self):
        module = load_package_module("SDXLQuantStatsCollector")
        session = {"layers": {}}
        backend = module.HSWQStatsCollectorBackend(session, threading.Lock(), "cpu")

        backend.hook_fn(
            None,
            torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2),
            torch.ones(2, 3, 2, 2),
            "conv",
        )
        backend.hook_fn(
            None,
            torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
            torch.ones(2, 4, 3),
            "linear3d",
        )
        backend.hook_fn(
            None,
            torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5),
            torch.ones(2, 5),
            "linear2d",
        )

        self.assertTrue(
            torch.equal(
                session["layers"]["conv"]["input_imp_sum"],
                torch.tensor([7.5, 11.5, 15.5], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                session["layers"]["linear3d"]["input_imp_sum"],
                torch.tensor([10.5, 11.5, 12.5], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                session["layers"]["linear2d"]["input_imp_sum"],
                torch.tensor([2.5, 3.5, 4.5, 5.5, 6.5], dtype=torch.float64),
            )
        )

    def test_snapshot_clones_tensor_data(self):
        module = load_package_module("SDXLQuantStatsCollector")
        session = {
            "meta": {"type": "hswq_dual_monitor_v2"},
            "layers": {
                "layer": {
                    "output_sum": 1.0,
                    "output_sq_sum": 2.0,
                    "out_count": 3,
                    "input_imp_sum": torch.tensor([1.0, 2.0], dtype=torch.float64),
                    "in_count": 4,
                }
            },
        }

        snapshot = module._snapshot_session_for_save(session)
        session["layers"]["layer"]["input_imp_sum"].add_(10)

        self.assertTrue(
            torch.equal(snapshot["layers"]["layer"]["input_imp_sum"], torch.tensor([1.0, 2.0], dtype=torch.float64))
        )

    def test_zit_get_session_sets_model_type_metadata(self):
        module = load_package_module("ZITQuantStatsCollector")
        module._SESSIONS.clear()
        module._SESSION_LOCKS.clear()

        session, ckpt_path, _ = module._get_session("zit_stats_test", "prefix", "session_a")

        self.assertEqual(session["meta"]["type"], "hswq_dual_monitor_v2")
        self.assertEqual(session["meta"]["model_type"], "NextDiT")
        self.assertTrue(ckpt_path.endswith("prefix_session_a.pt"))


if __name__ == "__main__":
    unittest.main()
