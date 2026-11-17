"""Versioned Track 3 scoring: legacy interpolation stays reproducible; support mode does not invent it."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import score_track3  # noqa: E402


@pytest.mark.parametrize("gap", [0, -1, float("nan"), float("inf")])
def test_support_rejects_invalid_gap(gap):
    with pytest.raises(ValueError, match="max_gap"):
        score_track3.temporal_support([0, 1], [0, 1], gap)


@pytest.mark.parametrize("times", [[0, 0, 1], [1, 0], [0, float("nan")]])
def test_support_rejects_ambiguous_reference_times(times):
    with pytest.raises(ValueError, match="timestamps"):
        score_track3.temporal_support([0, 1], times, .05)


def test_partial_blind_reference_does_not_claim_endpoint_error(tmp_path):
    t = np.arange(0., 2., .01)
    p = np.zeros((len(t), 3))
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_est(est, t, p)
    _write_gt(gt, t[t < .8], p[t < .8])
    result = score_track3.score(est, gt, {"t0_ns": 0, "t_fail_s": .5, "blind_s": .5}, mode="support")
    assert result["blind_rows"] == 51  # .5 through 1.0; later rows are not this blind window
    assert 0 < result["blind_support_coverage"] < 1
    assert result["blind_last_supported_m"] == pytest.approx(0)
    assert result["blind_end_supported"] is False
    assert "blind_end_m" not in result


def test_support_rejects_nonfinite_pose_instead_of_scoring_nan(tmp_path):
    t = np.arange(0., .1, .01)
    p = np.zeros((len(t), 3))
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_gt(gt, t, p)
    p[4, 0] = np.nan
    _write_est(est, t, p)
    with pytest.raises(ValueError, match="finite"):
        score_track3.score(est, gt, mode="support")


def _write_est(path, t, p, t_shift=0.0):
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    np.savetxt(path, np.column_stack([np.asarray(t, float) + t_shift, p, q]))


def _write_gt(path, t, p):
    lines = []
    for ti, xyz in zip(np.asarray(t, float), np.asarray(p, float)):
        ns = int(round(float(ti) * 1e9))
        lines.append(f"{ns},{xyz[0]},{xyz[1]},{xyz[2]},1,0,0,0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_imu(path, t):
    lines = [f"{int(round(float(ti) * 1e9))},0,0,0,0,0,9.81" for ti in t]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gap_case(tmp_path):
    """Reference jumps 10 m across a 2.49 s gap. The estimate stays at the origin."""
    gt_t = np.array([0.0, 0.01, 2.50])
    gt_p = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    te = np.concatenate([np.round(np.arange(0.0, 2.50, 0.01), 5), [2.50]])
    pe = np.zeros((len(te), 3))
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_est(est, te, pe)
    _write_gt(gt, gt_t, gt_p)
    return est, gt, te, pe, gt_t, gt_p


def test_legacy_mode_still_interpolates_a_multi_second_gap(tmp_path):
    est, gt, te, pe, gt_t, gt_p = _gap_case(tmp_path)
    legacy = score_track3.score(est, gt, mode="legacy")
    tg, pg = gt_t, gt_p
    inside = (te >= tg[0]) & (te <= tg[-1])
    te_i, pe_i = te[inside], pe[inside]
    pgi = np.stack([np.interp(te_i, tg, pg[:, i]) for i in range(3)], 1)
    rot, trans = score_track3.umeyama(pe_i, pgi)
    expected = float(np.sqrt(np.mean(np.sum((pe_i @ rot.T + trans - pgi) ** 2, 1))))
    assert legacy["schema"] == score_track3.SCHEMA_LEGACY
    assert legacy["mode"] == "legacy"
    assert legacy["ate_rmse_m"] == pytest.approx(expected, abs=1e-9)
    assert expected > 1.0
    support = score_track3.score(est, gt, mode="support")
    assert support["schema"] == score_track3.SCHEMA_SUPPORT
    assert support["gap_unsupported_rows"] > 100
    assert support["supported_rows"] == 3
    assert support["ate_rmse_m"] != pytest.approx(legacy["ate_rmse_m"], abs=1e-3)
    # The far exact sample is kept; the open gap is not filled in.
    assert support["support_coverage"] < 0.05


def test_support_mode_does_not_score_outside_the_reference_span(tmp_path):
    est, gt, te, pe, gt_t, gt_p = _gap_case(tmp_path)
    te2 = np.concatenate([[-1.0], te, [5.0]])
    pe2 = np.zeros((len(te2), 3))
    est2 = tmp_path / "est2.txt"
    _write_est(est2, te2, pe2)
    alone = score_track3.score(est, gt, mode="support")
    with_ends = score_track3.score(est2, gt, mode="support")
    assert with_ends["endpoint_unsupported_rows"] == 2
    assert with_ends["supported_rows"] == alone["supported_rows"]
    assert with_ends["ate_rmse_m"] == pytest.approx(alone["ate_rmse_m"], abs=1e-9)
    assert with_ends["support_coverage"] < alone["support_coverage"]


def test_exact_reference_sample_next_to_a_wide_gap_is_supported(tmp_path):
    est, gt, *_ = _gap_case(tmp_path)
    support = score_track3.score(est, gt, mode="support")
    te = np.concatenate([np.round(np.arange(0.0, 2.50, 0.01), 5), [2.50]])
    tg = np.array([0.0, 0.01, 2.50])
    mask, endpoint, gapped = score_track3.temporal_support(te, tg, 0.05)
    assert mask[0] and mask[1] and mask[-1]
    assert gapped[2]
    assert not endpoint[0]
    assert support["supported_rows"] == int(mask.sum())


def test_clean_support_score_has_no_blind_metrics_and_legacy_still_emits_them(tmp_path):
    t = np.arange(0.0, 1.0, 0.01)
    p = np.column_stack([t * 0.1, np.zeros(len(t)), np.zeros(len(t))])
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_est(est, t, p)
    _write_gt(gt, t, p)
    legacy = score_track3.score(est, gt, mode="legacy")
    support = score_track3.score(est, gt, mode="support")
    assert "blind_mean_m" in legacy and "blind_end_m" in legacy
    assert "blind_mean_m" not in support and "blind_end_m" not in support
    assert support["failure_declared"] is False
    assert support["support_coverage"] == 1.0
    assert support["ate_rmse_m"] == pytest.approx(0.0, abs=1e-8)


def test_declared_failure_without_reference_does_not_invent_a_blind_error(tmp_path):
    t = np.arange(0.0, 1.0, 0.01)
    p = np.zeros((len(t), 3))
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_est(est, t, p)
    _write_gt(gt, t[t <= 0.4], p[t <= 0.4])
    truth = {"t0_ns": 0, "t_fail_s": 0.5}
    support = score_track3.score(est, gt, truth, mode="support")
    assert support["failure_declared"] is True
    assert support["blind_supported_rows"] == 0
    assert support["blind_support_coverage"] == 0.0
    assert support["blind_status"] == "unsupported"
    assert "blind_mean_m" not in support and "blind_end_m" not in support


def test_declared_failure_with_support_reports_blind_error_and_coverage(tmp_path):
    t = np.arange(0.0, 1.0, 0.01)
    p = np.zeros((len(t), 3))
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_est(est, t, p)
    _write_gt(gt, t, p)
    truth = {"t0_ns": 0, "t_fail_s": 0.5, "dropouts_s": [[0.2, 0.25]]}
    support = score_track3.score(est, gt, truth, mode="support")
    assert support["blind_status"] == "supported"
    assert support["blind_support_coverage"] == 1.0
    assert support["blind_mean_m"] == pytest.approx(0.0, abs=1e-8)
    assert support["blind_end_m"] == pytest.approx(0.0, abs=1e-8)
    assert "dropout_m" in support
    assert support["dropout_support_coverage"] == 1.0


def test_timestamp_contract_uses_rtol_zero(tmp_path):
    # Unix-scale stamps: the old default rtol of 1e-5 would accept a 1 s miss.
    t = 1.5e9 + np.arange(0.0, 0.5, 0.01)
    p = np.zeros((len(t), 3))
    gt = tmp_path / "gt.csv"
    imu = tmp_path / "imu.csv"
    _write_gt(gt, np.concatenate([t, t + 1.0]), np.zeros((len(t) * 2, 3)))
    _write_imu(imu, t)
    near = tmp_path / "near.txt"
    far = tmp_path / "far.txt"
    _write_est(near, t, p, t_shift=1e-7)
    _write_est(far, t, p, t_shift=1.0)
    assert score_track3.score(near, gt, input_csv=imu, mode="support")["rows_match_input"] is True
    assert score_track3.score(far, gt, input_csv=imu, mode="support")["rows_match_input"] is False
    assert score_track3.score(far, gt, input_csv=imu, mode="legacy")["rows_match_input"] is False


def test_empty_ground_truth_fails_honestly(tmp_path):
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_est(est, [0.0, 0.01], np.zeros((2, 3)))
    gt.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no ground-truth"):
        score_track3.score(est, gt, mode="support")


def test_single_reference_sample_does_not_extrapolate(tmp_path):
    t = np.array([0.0, 0.5, 1.0])
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_est(est, t, np.zeros((3, 3)))
    _write_gt(gt, [0.5], [[1.0, 0.0, 0.0]])
    support = score_track3.score(est, gt, mode="support")
    assert support["supported_rows"] == 1
    assert support["endpoint_unsupported_rows"] == 2
    assert support["status"] == "insufficient_support"
    assert "ate_rmse_m" not in support


def test_cli_defaults_to_legacy_and_accepts_support(tmp_path):
    t = np.arange(0.0, 0.3, 0.01)
    p = np.zeros((len(t), 3))
    est, gt = tmp_path / "est.txt", tmp_path / "gt.csv"
    _write_est(est, t, p)
    _write_gt(gt, t, p)
    legacy = json.loads(
        __import__("subprocess").check_output(
            [sys.executable, str(ROOT / "tools" / "score_track3.py"), str(est), str(gt)], text=True))
    support = json.loads(
        __import__("subprocess").check_output(
            [sys.executable, str(ROOT / "tools" / "score_track3.py"), str(est), str(gt), "--mode", "support"],
            text=True))
    assert legacy["mode"] == "legacy" and "support_coverage" not in legacy
    assert support["mode"] == "support" and support["support_coverage"] == 1.0
