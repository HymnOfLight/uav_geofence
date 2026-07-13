from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:
    import z3
except ImportError as exc:  # pragma: no cover
    z3 = None
    _IMPORT_ERROR = exc

from .quantization import Int8MLP, QMAX, QMIN


def require_z3() -> None:
    if z3 is None:
        raise RuntimeError("z3-solver is required for SMT experiments: pip install z3-solver") from _IMPORT_ERROR


def round_div_z3(n, d: int):
    return z3.If(n >= 0, (n + d // 2) / d, -((-n + d // 2) / d))


def clamp_z3(x, lo: int, hi: int):
    return z3.If(x < lo, lo, z3.If(x > hi, hi, x))


@dataclass
class EncodedNetwork:
    inputs: list
    outputs: list
    layer_values: list[list]
    constraints: list


def encode_int8_network(net: Int8MLP, prefix: str = "qnn") -> EncodedNetwork:
    require_z3()
    inputs = [z3.Int(f"{prefix}_in_{i}") for i in range(net.weights[0].shape[1])]
    constraints = [z3.And(v >= QMIN, v <= QMAX) for v in inputs]
    h = inputs
    all_layers = [inputs]
    for li, (w, b) in enumerate(zip(net.weights, net.biases)):
        out = []
        for j in range(w.shape[0]):
            acc = int(b[j]) + z3.Sum([int(w[j, k]) * h[k] for k in range(w.shape[1])])
            rq = round_div_z3(acc, net.qscale)
            value = clamp_z3(rq, 0 if li < len(net.weights) - 1 else QMIN, QMAX)
            v = z3.Int(f"{prefix}_l{li}_o{j}")
            constraints.append(v == value)
            out.append(v)
        h = out
        all_layers.append(out)
    return EncodedNetwork(inputs, h, all_layers, constraints)


def verify_fixed_input(net: Int8MLP, qinput: np.ndarray, timeout_ms: int = 3000) -> dict:
    enc = encode_int8_network(net, "fixed")
    expected = net.forward_q(np.asarray(qinput, dtype=np.int16)).astype(int)
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(enc.constraints)
    solver.add([v == int(x) for v, x in zip(enc.inputs, qinput)])
    solver.add(z3.Or([v != int(x) for v, x in zip(enc.outputs, expected)]))
    t0 = time.perf_counter()
    ans = solver.check()
    elapsed = time.perf_counter() - t0
    return {
        "consistent": ans == z3.unsat,
        "solver_status": str(ans),
        "elapsed_s": elapsed,
        "qinput": np.asarray(qinput, dtype=int).tolist(),
        "expected": expected.tolist(),
    }


def verify_action_halfspace(
    net: Int8MLP,
    input_lo: np.ndarray,
    input_hi: np.ndarray,
    normal: np.ndarray,
    min_outward_q: int,
    timeout_ms: int,
    prefix: str = "cell",
) -> dict:
    """Prove normal dot q_action >= min_outward_q over an integer input box."""
    enc = encode_int8_network(net, prefix)
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(enc.constraints)
    for v, lo, hi in zip(enc.inputs, input_lo, input_hi):
        solver.add(v >= int(lo), v <= int(hi))
    coeff = np.asarray(normal, dtype=int)
    lhs = z3.Sum([int(coeff[i]) * enc.outputs[i] for i in range(2)])
    solver.add(lhs < int(min_outward_q))
    t0 = time.perf_counter()
    ans = solver.check()
    elapsed = time.perf_counter() - t0
    result = {"elapsed_s": elapsed, "solver_status": str(ans)}
    if ans == z3.unsat:
        result["status"] = "SAFE"
    elif ans == z3.unknown:
        result["status"] = "UNKNOWN"
        result["reason"] = solver.reason_unknown()
    else:
        model = solver.model()
        result["status"] = "UNSAFE"
        result["counterexample_qinput"] = [model.eval(v, model_completion=True).as_long() for v in enc.inputs]
        result["counterexample_qoutput"] = [model.eval(v, model_completion=True).as_long() for v in enc.outputs]
    return result

