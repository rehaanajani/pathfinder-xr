"""Fuzz: random overlapping fault schedules through the shipped settings (ours_cfg).

Invariants on every sample: no exception, finite output, covariance symmetric and PSD,
StreamingTracker == filter.run (position and covariance, 1e-9), faster than real time.
Replay a failure:  STRESS_CASE=<i> python -m pytest -q tests/stress/test_fuzz.py -k single
"""
import os
from multiprocessing import Pool

import pytest

from fuzz import random_case
from harness import check_invariants, repro, run_pair

N_FAST = int(os.environ.get("STRESS_FUZZ_N", "40"))
N_SLOW = int(os.environ.get("STRESS_FUZZ_N", "600"))


def _one(i):
    case = random_case(i)
    try:
        return i, check_invariants(run_pair(case, check_every=5)), case
    except Exception as e:                                   # noqa: BLE001
        return i, [f"EXCEPTION {type(e).__name__}: {e}"], case


def _report(results):
    fails = [(i, b, c) for i, b, c in results if b]
    msg = "\n".join(f"fuzz case {i}: {b}\n   {repro(c)}" for i, b, c in fails[:10])
    assert not fails, f"{len(fails)}/{len(results)} fuzz cases violated an invariant\n{msg}"


def test_fuzz_fast():
    _report([_one(i) for i in range(N_FAST)])


def test_fuzz_single():
    i = int(os.environ.get("STRESS_CASE", "0"))
    _report([_one(i)])


@pytest.mark.slow
def test_fuzz_600_cases():
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        res = pool.map(_one, range(N_SLOW))
    _report(res)


@pytest.mark.slow
def test_fuzz_with_learned_model():
    """Same invariants with the real displacement network running (TLIO-trained weights on synthetic
    head motion are out of distribution, which is the point: the filter must stay sane anyway)."""
    from pathfinder.predictor import Predictor
    import torch
    torch.set_num_threads(1)
    if not os.path.exists("models/dispnet_faults/best.pt"):
        pytest.skip("model not downloaded")
    out = []
    for i in range(60):
        case = random_case(5000 + i, duration=15.0)
        try:
            r = run_pair(case, predictor_factory=lambda: Predictor("models/dispnet_faults/best.pt"), check_every=10)
            out.append((i, check_invariants(r), case))
        except Exception as e:                               # noqa: BLE001
            out.append((i, [f"EXCEPTION {type(e).__name__}: {e}"], case))
    _report(out)
