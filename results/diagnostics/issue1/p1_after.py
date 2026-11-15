"""After-version of the audit probe p1 (findings 2, 3, 4) for the new monitor API. Synthetic only."""
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'tests')
from pathfinder.faults import apply_imu_faults, scenarios
from pathfinder.imu_health import STATE_NAMES, ImuGeometry, ImuMonitor
from pathfinder.pipeline import fuse_imus
from pathfinder.synthetic import head_walk
from test_imu_health import _rigid_offsets

print("== 1. IMU quality passed to the filter (3 IMUs): per-window state histogram and extra process noise, 20-30 s")
for name in ["clean", "imu_noise_all", "one_imu_bias", "one_imu_stuck", "blackout_plus_vibration_all", "blackout_plus_spikes_all",
             "blackout_plus_saturation_all", "blackout_plus_dropout_all", "dark_plus_imu_drift"]:
    s = head_walk(60, n_imus=3, seed=1)
    bad = apply_imu_faults(s, scenarios(10.0)[name], 0)
    fg, fa, info = fuse_imus(bad)
    ws = info["window_state"][200:300]
    hist = {STATE_NAMES[k]: int((ws == k).sum()) for k in np.unique(ws)}
    print(f"{name:30s} states {hist} | q_add_a mean {info['q_add_a'][4000:6000].mean():.3g} | dropped per IMU {(info['weights'][:, 200:300] == 0).mean(1).round(2)}")

print("== 2. false alarms on CLEAN data, nominal p_fa=1e-4 (tested = calibrated windows)")
for K in (3, 5):
    al = te = 0
    for seed in range(8):
        s = head_walk(60, n_imus=K, seed=seed)
        _, _, info = ImuMonitor(K).run_batch(s.t, s.gyro, s.acc)
        c = info["calibrated"]
        al += int(info["detected"][c].sum())
        te += int(c.sum())
    print(f"K={K}: {al} alarms / {te} windows = {al / te:.5f}")

print("== 3. lever arms, no faults: share of windows not NOMINAL, without vs with ImuGeometry")
for arm in (0.02, 0.05, 0.10):
    arms = [[arm, 0, 0], [-arm, 0, 0], [0, arm, 0]]
    a0 = a1 = n = 0
    for seed in range(8):
        s = _rigid_offsets(head_walk(60, n_imus=3, seed=seed), arms)
        _, _, i0 = ImuMonitor(3).run_batch(s.t, s.gyro, s.acc)
        _, _, i1 = ImuMonitor(3, geometry=ImuGeometry(np.repeat(np.eye(3)[None], 3, 0), np.array(arms))).run_batch(s.t, s.gyro, s.acc)
        c = i1["calibrated"]
        a0 += int((i0["window_state"][c] != 1).sum())
        a1 += int((i1["window_state"][c] != 1).sum())
        n += int(c.sum())
    print(f"+-{arm * 100:.0f} cm: co-located assumption {a0 / n:.3f} | with geometry {a1 / n:.4f}")
