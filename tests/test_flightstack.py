from __future__ import annotations

import csv
import dataclasses
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geofence_qnn.config import DataConfig, load_config
from geofence_qnn.controller import TeacherController
from geofence_qnn.data import make_dataset_from_config
from geofence_qnn.dynamics import step_dynamics
from geofence_qnn.flightstack import (
    ArduPilotFenceTeacher,
    PX4GeofenceTeacher,
    load_csv_log,
    make_teacher,
    trajectories_to_dataset,
)
from geofence_qnn.features import batch_state_features, state_features
from geofence_qnn.flightstack.logs import (
    Trajectory,
    _ulog_to_trajectory,
    align_trajectories,
    download_log,
    make_flight_log_dataset,
)
from geofence_qnn.geometry import ForbiddenBox
from geofence_qnn.model import MLP
from geofence_qnn.quantization import Int8MLP
from geofence_qnn.simulation import run_monte_carlo

try:
    from pymavlink.dialects.v20 import common as mavlink2

    HAVE_PYMAVLINK = True
except ImportError:  # pragma: no cover
    HAVE_PYMAVLINK = False

BOX = ForbiddenBox(-5.0, 5.0, -12.0, 12.0)
GOAL = np.array([60.0, 0.0])
AMAX, VMAX, MARGIN, SCALE = 4.0, 8.0, 1.0, 50.0


class TeacherBackendTests(unittest.TestCase):
    def test_factory_backends(self):
        self.assertIsInstance(make_teacher("builtin"), TeacherController)
        self.assertIsInstance(make_teacher("px4", vmax=VMAX, amax=AMAX), PX4GeofenceTeacher)
        self.assertIsInstance(make_teacher("ardupilot", vmax=VMAX, amax=AMAX), ArduPilotFenceTeacher)
        with self.assertRaises(ValueError):
            make_teacher("betaflight")

    def test_factory_param_override(self):
        teacher = make_teacher("px4", {"gf_predict": False}, vmax=VMAX, amax=AMAX)
        self.assertFalse(teacher.gf_predict)
        self.assertAlmostEqual(teacher.mpc_xy_vel_max, VMAX)

    def _check_teacher(self, teacher):
        # Actions are always saturated to the actuator limit.
        rng = np.random.default_rng(0)
        for _ in range(200):
            state = np.r_[rng.uniform(-60, 60, 2), rng.uniform(-VMAX, VMAX, 2)]
            u = teacher.action(state, GOAL, BOX, AMAX, MARGIN)
            self.assertTrue(np.all(np.abs(u) <= AMAX + 1e-9))
        # Flying straight at the left fence face: commanded acceleration must
        # push outward (negative x is outward for that face).
        approach = np.array([-7.5, 0.0, 4.0, 0.0])
        u = teacher.action(approach, GOAL, BOX, AMAX, MARGIN)
        normal = BOX.nearest_outward_normal(approach[:2])
        self.assertGreater(float(np.dot(u, normal)), 0.0)
        # Inside the safety margin the teacher must push back out.
        inside = np.array([-5.5, 0.0, 0.0, 0.0])
        u_in = teacher.action(inside, GOAL, BOX, AMAX, MARGIN)
        self.assertGreater(float(np.dot(u_in, BOX.nearest_outward_normal(inside[:2]))), 0.0)

    def test_px4_behavior(self):
        self._check_teacher(make_teacher("px4", vmax=VMAX, amax=AMAX))

    def test_ardupilot_behavior(self):
        self._check_teacher(make_teacher("ardupilot", vmax=VMAX, amax=AMAX))

    def _closed_loop_safe(self, teacher):
        state = np.array([-30.0, 0.0, 5.0, 0.0])
        for _ in range(600):
            u = teacher.action(state, GOAL, BOX, AMAX, MARGIN)
            state = step_dynamics(state, u, 0.05, VMAX, AMAX)
            self.assertFalse(BOX.contains(state[:2]), f"fence crossed at {state}")

    def test_px4_closed_loop_never_crosses(self):
        self._closed_loop_safe(make_teacher("px4", vmax=VMAX, amax=AMAX))

    def test_ardupilot_closed_loop_never_crosses(self):
        self._closed_loop_safe(make_teacher("ardupilot", vmax=VMAX, amax=AMAX))


def _simulate_csv_log(path: Path, episodes: int = 3, steps: int = 300):
    """Roll out the builtin teacher and store the flights as a CSV log."""
    teacher = TeacherController()
    rng = np.random.default_rng(1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "t", "x", "y", "vx", "vy", "ax", "ay"])
        for episode in range(episodes):
            state = np.r_[rng.uniform(-40, -20), rng.uniform(-15, 15), rng.uniform(0, 3), 0.0]
            for k in range(steps):
                u = teacher.action(state, GOAL, BOX, AMAX, MARGIN)
                writer.writerow([episode, k * 0.05, *state, *u])
                state = step_dynamics(state, u, 0.05, VMAX, AMAX)


class FlightLogTests(unittest.TestCase):
    def test_csv_roundtrip_to_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "flights.csv"
            _simulate_csv_log(log)
            trajectories = load_csv_log(log)
            self.assertEqual(len(trajectories), 3)
            states, x, y = trajectories_to_dataset(
                trajectories, 0.05, GOAL, BOX, SCALE, VMAX, AMAX, MARGIN
            )
            self.assertEqual(x.shape[1], 6)
            self.assertEqual(y.shape[1], 2)
            self.assertEqual(len(states), len(x))
            self.assertTrue(np.all(np.abs(y) <= 1.0 + 1e-9))
            for s in states:
                self.assertFalse(BOX.contains(s[:2], margin=MARGIN))

    def test_make_flight_log_dataset_glob_and_subsample(self):
        with tempfile.TemporaryDirectory() as tmp:
            _simulate_csv_log(Path(tmp) / "a.csv", episodes=2, steps=200)
            _simulate_csv_log(Path(tmp) / "b.csv", episodes=1, steps=200)
            states, x, y = make_flight_log_dataset(
                [f"{tmp}/*.csv"], "csv", 100, 7, 0.05, GOAL, BOX, SCALE, VMAX, AMAX, MARGIN
            )
            self.assertEqual(len(x), 100)
            self.assertEqual(len(states), 100)
            self.assertEqual(y.shape, (100, 2))

    def test_missing_logs_raise(self):
        with self.assertRaises(FileNotFoundError):
            make_flight_log_dataset(
                ["/nonexistent/*.csv"], "csv", 10, 0, 0.05, GOAL, BOX, SCALE, VMAX, AMAX, MARGIN
            )

    def test_ulog_conversion_ned_mapping(self):
        class StubDataset:
            def __init__(self, data):
                self.data = data

        class StubULog:
            def __init__(self, data):
                self._data = data

            def get_dataset(self, topic):
                assert topic == "vehicle_local_position"
                return StubDataset(self._data)

        n = 20
        data = {
            "timestamp": (np.arange(n) * 5e4).astype(np.uint64),  # 20 Hz in us
            "x": np.full(n, 1.0),  # north
            "y": np.full(n, 2.0),  # east
            "vx": np.full(n, 3.0),  # v_north
            "vy": np.full(n, 4.0),  # v_east
            "ax": np.full(n, 0.5),
            "ay": np.full(n, 0.25),
        }
        traj = _ulog_to_trajectory(StubULog(data), "vehicle_local_position", "ned", (0.0, 0.0), "stub")
        # NED -> world: x = east, y = north (same for velocity/acceleration).
        np.testing.assert_allclose(traj.pos[0], [2.0, 1.0])
        np.testing.assert_allclose(traj.vel[0], [4.0, 3.0])
        np.testing.assert_allclose(traj.acc[0], [0.25, 0.5])
        self.assertAlmostEqual(traj.t[1] - traj.t[0], 0.05)

    @unittest.skipUnless(HAVE_PYMAVLINK, "pymavlink not installed")
    def test_ardupilot_tlog_parsing(self):
        from geofence_qnn.flightstack import load_ardupilot_log

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flight.tlog"
            with path.open("wb") as f:
                mav = mavlink2.MAVLink(f, srcSystem=1, srcComponent=1)
                for k in range(40):
                    msg = mav.local_position_ned_encode(
                        k * 50, 10.0 + 0.1 * k, 20.0 + 0.2 * k, -10.0, 0.1, 0.2, 0.0
                    )
                    f.write(struct.pack(">Q", k * 50000) + msg.pack(mav))
            trajectories = load_ardupilot_log(path)
            self.assertEqual(len(trajectories), 1)
            traj = trajectories[0]
            self.assertEqual(len(traj), 40)
            # NED (north=10.., east=20..) -> world (x=east, y=north).
            np.testing.assert_allclose(traj.pos[0], [20.0, 10.0])
            np.testing.assert_allclose(traj.vel[0], [0.2, 0.1])

    @unittest.skipUnless(HAVE_PYMAVLINK, "pymavlink not installed")
    def test_sitl_recorder_against_fake_vehicle(self):
        import threading
        import time

        from pymavlink import mavutil

        from geofence_qnn.flightstack.sitl import record_sitl_trajectories

        port = 14760
        stop = threading.Event()

        def fake_vehicle():
            out = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}", source_system=1)
            k = 0
            while not stop.is_set():
                out.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_QUADROTOR,
                    mavutil.mavlink.MAV_AUTOPILOT_PX4,
                    0, 0, mavutil.mavlink.MAV_STATE_ACTIVE,
                )
                out.mav.local_position_ned_send(
                    k * 50, 1.0 + 0.01 * k, 2.0 + 0.02 * k, -10.0, 0.2, 0.4, 0.0
                )
                k += 1
                time.sleep(0.05)
            out.close()

        thread = threading.Thread(target=fake_vehicle, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_csv = Path(tmp) / "sitl.csv"
                summary = record_sitl_trajectories(
                    f"udp:127.0.0.1:{port}",
                    out_csv,
                    episodes=1,
                    duration_s=1.5,
                    rate_hz=20.0,
                    goal=(60.0, 0.0),
                    heartbeat_timeout_s=10.0,
                )
                self.assertGreater(summary["samples"], 5)
                trajectories = load_csv_log(out_csv)
                self.assertEqual(len(trajectories), 1)
                traj = trajectories[0]
                # NED north=1.., east=2.. -> world x=east, y=north.
                np.testing.assert_allclose(traj.pos[0], [2.0, 1.0], atol=0.5)
                np.testing.assert_allclose(traj.vel[0], [0.4, 0.2], atol=1e-6)
        finally:
            stop.set()
            thread.join(timeout=2.0)

    def test_interleaved_and_bad_timestamps_are_cleaned(self):
        t = np.array([0.0, 0.1, 0.1, 0.05, np.nan, 0.2])
        base = np.zeros(6)
        from geofence_qnn.flightstack.logs import _finalize_trajectory

        cleaned = _finalize_trajectory(t, base, base, base, base, None, None, "xy", (0.0, 0.0), "s")
        self.assertTrue(np.all(np.diff(cleaned.t) > 0))


class VectorizationTests(unittest.TestCase):
    def test_batch_features_match_per_sample(self):
        rng = np.random.default_rng(11)
        states = np.column_stack(
            [rng.uniform(-80, 80, 50), rng.uniform(-80, 80, 50), rng.uniform(-VMAX, VMAX, 50), rng.uniform(-VMAX, VMAX, 50)]
        )
        batch = batch_state_features(states, GOAL, BOX, SCALE, VMAX)
        for i, s in enumerate(states):
            np.testing.assert_allclose(batch[i], state_features(s, GOAL, BOX, SCALE, VMAX))

    def test_contains_batch_matches_scalar(self):
        rng = np.random.default_rng(12)
        points = rng.uniform(-20, 20, size=(200, 2))
        batch = BOX.contains_batch(points, margin=MARGIN)
        for i, p in enumerate(points):
            self.assertEqual(bool(batch[i]), BOX.contains(p, margin=MARGIN))


class RealLogHelpersTests(unittest.TestCase):
    def test_auto_align_centers_flights_on_fence(self):
        t = np.arange(10) * 0.1
        pos = np.column_stack([np.linspace(1000.0, 1060.0, 10), np.linspace(-500.0, -460.0, 10)])
        traj = Trajectory(t=t, pos=pos, vel=np.zeros((10, 2)), acc=np.ones((10, 2)))
        aligned = align_trajectories([traj], BOX)[0]
        bbox_center = 0.5 * (aligned.pos.min(axis=0) + aligned.pos.max(axis=0))
        np.testing.assert_allclose(bbox_center, BOX.center, atol=1e-9)
        # Translation only: dynamics quantities unchanged.
        np.testing.assert_allclose(aligned.vel, traj.vel)
        np.testing.assert_allclose(aligned.acc, traj.acc)
        np.testing.assert_allclose(aligned.t, traj.t)

    def test_url_logs_download_and_cache(self):
        import http.server
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            served = Path(tmp) / "served"
            served.mkdir()
            _simulate_csv_log(served / "flights.csv", episodes=1, steps=50)
            handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(served), **kw)
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{server.server_address[1]}/flights.csv"
            cache = Path(tmp) / "cache"
            try:
                first = download_log(url, "csv", cache_dir=cache)
                self.assertTrue(first.exists())
                self.assertEqual(len(load_csv_log(first)), 1)
            finally:
                server.shutdown()
            # Second call must hit the cache: the server is already down.
            second = download_log(url, "csv", cache_dir=cache)
            self.assertEqual(first, second)

    def test_failed_download_raises_and_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            with self.assertRaises(RuntimeError):
                download_log("http://127.0.0.1:9/missing.csv", "csv", cache_dir=cache)
            self.assertEqual(list(cache.glob("*")), [])

    def test_fetch_script_duration_parser(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "fetch_px4_logs", Path(__file__).parents[1] / "scripts/fetch_px4_logs.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.parse_duration_s("21s"), 21)
        self.assertEqual(module.parse_duration_s("3m17s"), 197)
        self.assertEqual(module.parse_duration_s("11m7s"), 667)
        self.assertEqual(module.parse_duration_s("1h2m3s"), 3723)
        self.assertEqual(module.parse_duration_s("garbage"), 0)


class ConfigAndPipelineTests(unittest.TestCase):
    def test_legacy_config_defaults(self):
        cfg = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
        self.assertEqual(cfg.data.source, "synthetic")
        self.assertEqual(cfg.teacher.backend, "builtin")
        self.assertEqual(cfg.simulation.controllers, ("teacher", "float", "int8", "int8_shield"))

    def test_flightstack_config(self):
        cfg = load_config(Path(__file__).parents[1] / "configs/smoke_flightstack.yaml")
        self.assertEqual(cfg.teacher.backend, "px4")
        self.assertIn("ardupilot", cfg.simulation.controllers)

    def test_dataset_from_csv_config(self):
        base = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "flights.csv"
            _simulate_csv_log(log, episodes=2, steps=200)
            cfg = dataclasses.replace(
                base,
                training=dataclasses.replace(base.training, samples=150),
                data=DataConfig(source="csv", logs=(str(log),), synthetic_fraction=0.2),
            )
            states, x, y = make_dataset_from_config(cfg)
            self.assertEqual(len(x), 150)
            self.assertEqual(y.shape[1], 2)
            self.assertTrue(np.all(np.abs(y) <= 1.0 + 1e-9))

    def test_monte_carlo_with_flightstack_baselines(self):
        model = MLP(weights=[np.zeros((2, 6))], biases=[np.zeros(2)])
        qnet = Int8MLP.from_float(model, 32)
        rows = run_monte_carlo(
            ["teacher", "px4", "ardupilot", "int8"],
            4,
            np.array([-30.0, -5.0, 1.0, -0.5]),
            np.array([-25.0, 5.0, 3.0, 0.5]),
            GOAL,
            BOX,
            MARGIN,
            0.05,
            0.01,
            VMAX,
            AMAX,
            SCALE,
            50,
            0.0,
            0.0,
            10,
            9,
            model,
            qnet,
            teacher=make_teacher("px4", vmax=VMAX, amax=AMAX),
        )
        self.assertEqual([r["controller"] for r in rows], ["teacher", "px4", "ardupilot", "int8"])
        for row in rows[:3]:
            self.assertEqual(row["violations"], 0)


if __name__ == "__main__":
    unittest.main()
