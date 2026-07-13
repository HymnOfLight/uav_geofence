from __future__ import annotations

import unittest

import numpy as np

from geofence_qnn.dynamics import interval_step, step_dynamics
from geofence_qnn.geometry import ForbiddenBox
from geofence_qnn.model import MLP
from geofence_qnn.quantization import Int8MLP, round_div_int
from geofence_qnn.smt import verify_fixed_input


class GeometryTests(unittest.TestCase):
    def test_clearance_sign(self):
        box = ForbiddenBox(-1, 1, -2, 2)
        self.assertAlmostEqual(box.clearance(np.array([-3.0, 0.0])), 2.0)
        self.assertLess(box.clearance(np.array([0.0, 0.0])), 0.0)
        self.assertTrue(box.contains(np.array([1.5, 0.0]), margin=0.5))

    def test_interval_intersection(self):
        box = ForbiddenBox(-1, 1, -1, 1)
        self.assertFalse(box.interval_may_intersect(np.array([-3, -0.5]), np.array([-2, 0.5])))
        self.assertTrue(box.interval_may_intersect(np.array([-2, -0.5]), np.array([0, 0.5])))


class DynamicsTests(unittest.TestCase):
    def test_point_is_inside_interval(self):
        state = np.array([0.0, 0.0, 1.0, -0.5])
        action = np.array([0.2, 0.1])
        point = step_dynamics(state, action, 0.05, 8.0, 4.0)
        lo, hi = interval_step(state, state, action, action, 0.05, 8.0, 0.0)
        self.assertTrue(np.all(point >= lo - 1e-12))
        self.assertTrue(np.all(point <= hi + 1e-12))


class QuantizationTests(unittest.TestCase):
    def test_round_div(self):
        n = np.array([-49, -48, -16, 15, 16, 48, 49])
        np.testing.assert_array_equal(round_div_int(n, 32), np.array([-2, -2, -1, 0, 1, 2, 2]))

    def test_smt_fixed_input(self):
        model = MLP(
            weights=[np.array([[0.5, -0.25], [-0.5, 0.75]]), np.array([[0.4, -0.3], [0.2, 0.6]])],
            biases=[np.array([0.1, -0.2]), np.array([0.0, 0.05])],
        )
        qnet = Int8MLP.from_float(model, qscale=32)
        for qin in [np.array([0, 0]), np.array([16, -8]), np.array([-32, 31])]:
            result = verify_fixed_input(qnet, qin, timeout_ms=3000)
            self.assertTrue(result["consistent"], result)


if __name__ == "__main__":
    unittest.main()

