from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from geofence_qnn.config import load_config
from geofence_qnn.data import make_dataset
from geofence_qnn.geometry import box_from_tuple
from geofence_qnn.model import MLP, train_mlp
from geofence_qnn.quantization import Int8MLP
from geofence_qnn.reachability import adaptive_reachability


class PipelineTests(unittest.TestCase):
    def test_training_and_quantization(self):
        cfg = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
        e = cfg.environment
        _, x, y = make_dataset(
            300,
            7,
            np.array(e.world_min),
            np.array(e.world_max),
            e.vmax,
            e.amax,
            np.array(e.goal),
            box_from_tuple(e.forbidden_box),
            e.position_scale,
            e.safety_margin,
        )
        model = MLP.create([6, 8, 8, 2], 7)
        before = float(np.mean((model.forward(x) - y) ** 2))
        train_mlp(model, x, y, 12, 64, 0.003, 0.0, 7)
        after = float(np.mean((model.forward(x) - y) ** 2))
        self.assertLess(after, before)
        qnet = Int8MLP.from_float(model, 32)
        self.assertEqual(qnet.forward(x[:4]).shape, (4, 2))

    def test_far_box_reachability_safe(self):
        model = MLP(
            weights=[np.zeros((2, 6))],
            biases=[np.zeros(2)],
        )
        qnet = Int8MLP.from_float(model, 32)
        results = adaptive_reachability(
            qnet,
            np.array([-50.0, -2.0, 0.0, 0.0]),
            np.array([-49.0, 2.0, 0.0, 0.0]),
            np.array([60.0, 0.0]),
            box_from_tuple((-5.0, 5.0, -10.0, 10.0)),
            50.0,
            8.0,
            4.0,
            1.0,
            0.05,
            0.0,
            10,
            1,
        )
        self.assertTrue(all(r.status == "SAFE" for r in results))


if __name__ == "__main__":
    unittest.main()

