import numpy as np
from scipy.stats import chi2

from pathfinder.scoring import reference_at, recovery_time
from tools.calibrate_vo import scale_score, gyro_rotation_check


def test_exact_reference_sample_is_supported_next_to_gap():
    times = np.array([0., .01, 1.])
    _, _, mask = reference_at(times, np.zeros((3, 3)), np.tile([0, 0, 0, 1.], (3, 1)),
                              np.array([.005, .5, 1.]))
    assert mask.tolist() == [True, False, True]


def test_sparse_samples_do_not_prove_sustained_recovery():
    assert recovery_time(np.array([0., 2., 4.]), np.zeros(3), 0, .1, hold=2) is None
    times = np.arange(0, 3, .01)
    assert recovery_time(times, np.zeros(len(times)), 0, .1, hold=2) == 0


def test_scale_selection_does_not_use_mean_alone():
    matched_quantiles = {"vo_nis_mean": 50., "vo_nis_median": chi2.ppf(.5, 6),
                         "vo_nis_p95": chi2.ppf(.95, 6)}
    matched_mean = {"vo_nis_mean": 6., "vo_nis_median": 1., "vo_nis_p95": 3.}
    assert scale_score(matched_quantiles) < scale_score(matched_mean)


def test_empty_gyro_diagnostic_has_no_fabricated_statistic():
    assert gyro_rotation_check(None, [], 0, 1)["n"] == 0
