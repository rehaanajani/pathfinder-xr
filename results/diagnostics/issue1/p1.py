import sys, numpy as np
sys.path.insert(0, '.')
from pathfinder.synthetic import head_walk
from pathfinder.faults import Fault, apply_imu_faults, scenarios
from pathfinder.pipeline import fuse_imus
from pathfinder.parity import ParityFDI, lever_arm_compensate
from scipy.spatial.transform import Rotation as R

print("== 1. imu confidence passed to the filter (3 IMUs, parity path)")
for name in ["clean","imu_noise_all","one_imu_bias","one_imu_stuck","blackout_plus_vibration_all","blackout_plus_spikes_all","blackout_plus_saturation_all","blackout_plus_dropout_all","dark_plus_imu_drift"]:
    s = head_walk(60, n_imus=3, seed=1)
    bad = apply_imu_faults(s, scenarios(10.0)[name], 0)
    fg, fa, info = fuse_imus(bad)
    c = info["confidence"]; w = info["weights"]
    det = (info["parity_D"] > ParityFDI(3).threshold)
    fw = slice(200, 300)  # fault window 20-30 s in 0.1 s windows
    print(f"{name:30s} conf min={c.min():.2f} mean={c.mean():.3f} | detected: pre-fault {det[10:200].mean():.3f} in-fault {det[fw].mean():.3f} | dropped per IMU in-fault {(w[:,fw]==0).mean(1).round(2)}")

print("== 2. parity false alarms on CLEAN data, nominal p_fa=1e-4")
for K in (3,5):
    fa=[]
    for seed in range(8):
        s = head_walk(60, n_imus=K, seed=seed)
        w, D, info = ParityFDI(K).run(s.gyro, s.acc, s.rate)
        fa.append(info["detected"][10:].mean())
    print(f"K={K} ideal co-located copies: false-alarm rate per window mean={np.mean(fa):.4f} max={np.max(fa):.4f}")

print("== 3. same, but IMUs physically offset (lever arms) -- not compensated in fuse_imus")
def with_lever(s, arms):
    out = s.copy()
    w = s.gyro[0]*0  # use noise-free truth rate from quaternions
    rot = R.from_quat(s.q)
    wb = np.zeros_like(s.p); wb[:-1] = (rot[:-1].inv()*rot[1:]).as_rotvec()/s.dt; wb[-1]=wb[-2]
    wd = np.gradient(wb, s.dt, axis=0)
    for k, r in enumerate(arms):
        r = np.asarray(r)
        out.acc[k] = s.acc[k] + np.cross(wb, np.cross(wb, r)) + np.cross(wd, r)
    return out
for arm in (0.02, 0.05, 0.10):
    arms = [[arm,0,0],[-arm,0,0],[0,arm,0]]
    fa=[]; drop=[]
    for seed in range(8):
        s = with_lever(head_walk(60, n_imus=3, seed=seed), arms)
        w, D, info = ParityFDI(3).run(s.gyro, s.acc, s.rate)
        fa.append(info["detected"][10:].mean()); drop.append((w[:,10:]==0).any(0).mean())
    print(f"lever arm ±{arm*100:.0f} cm, no faults: windows flagged faulty mean={np.mean(fa):.3f} max={np.max(fa):.3f}")
