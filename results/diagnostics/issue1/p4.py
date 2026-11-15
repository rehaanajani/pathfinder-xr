import sys, json, numpy as np
sys.path.insert(0, '.')
from dataclasses import replace
from pathfinder.synthetic import head_walk
from pathfinder.faults import Fault, scenarios
from pathfinder import pipeline
from pathfinder.filter import run, FilterConfig
B = json.load(open('results/summary_before_upgrades.json'))["scenarios"]; A = json.load(open('results/summary.json'))["scenarios"]
for n in ("camera_wrong_fix","dark_plus_imu_drift","clean","camera_blackout"):
    print(n, "TLIO test, before upgrades: naive %.3f ours %.3f | committed now: naive %.3f ours %.3f" % (B[n]["naive_fusion"]["ate_m"], B[n]["ours"]["ate_m"], A[n]["naive_fusion"]["ate_m"], A[n]["ours"]["ate_m"]))
print("== wrong-fix scenario (benchmark default: offset [0.6,-0.4,0], 10 s), synthetic walk, oracle displacement as in tests/test_filter.py")
def oracle(s):
    def predictor(win):
        k = predictor.k; predictor.k += 20
        return s.p[k] - s.p[k-200], np.eye(3)*0.05**2
    predictor.k, predictor.window, predictor.every = 200, 200, 20
    return predictor
cfgs = {"naive (no health, no gate)": FilterConfig(use_health=False, use_gate=False),
        "FilterConfig() as used by the unit test": FilterConfig(),
        "ours_cfg() as used by the benchmark": pipeline.ours_cfg(),
        "ours_cfg() with adaptive_q=False": replace(pipeline.ours_cfg(), adaptive_q=False)}
for lab, cfg in cfgs.items():
    for use_model in (True, False):
        e=[]; rec=[]
        for seed in range(4):
            s = head_walk(60, n_imus=3, seed=seed)
            bad, cam = pipeline.prepare(s, scenarios(10.0)["camera_wrong_fix"], seed)
            fg, fa, info = pipeline.fuse_imus(bad)
            est, log = run(s, fg, fa, None, cam, oracle(s) if use_model else None, replace(cfg))
            err = np.linalg.norm(est - s.p, axis=1)
            e.append(err[int(21*200):int(30*200)].mean()); rec.append(log.cam_recovered)
        print(f"{lab:42s} {'oracle model' if use_model else 'no model    '}: mean error while camera is wrong {np.mean(e):.3f} m (offset injected is 0.72 m); re-anchors {rec}")
