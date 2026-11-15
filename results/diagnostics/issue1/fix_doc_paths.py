from pathlib import Path

d = Path("docs/ESTIMATOR.md")
s = d.read_text(encoding="utf-8")
reps = [
    ("baseline_evidence/", "results/diagnostics/issue1/evidence/"),
    ("`probes/", "`results/diagnostics/issue1/"),
    ("(and is not inside a provisional period after an earlier forced\n   re-anchor)",
     "(and is not inside the provisional period that follows any re-anchor which ended a jump\n"
     "   episode, `provisional_factor` x the jump hold-out)"),
    ("""the estimate is withdrawn
(`valid` False) once its std passes 1 m. **With the models-v1 network the covariance is still
overconfident during camera outages on synthetic motion** (≈ 30–50 % inside the 99.7 % region): the network's
claimed variance, even ×10 (a stand-in for overlap correlation) or with cloning, does not cover its error on
this out-of-domain motion. This finding is open; see below.""",
     """the estimate is withdrawn
(`valid` False) once its std passes 1 m. **With the models-v1 network the covariance is overconfident during
camera outages on synthetic motion**: only 29–41 % of outage samples fall inside the 99.7 % region, and the
network's claimed variance, even ×10 (a stand-in for overlap correlation) or with cloning, does not cover its
error on this out-of-domain motion. Mitigation, not a fix: while the camera does not anchor the state and the
self-check has measured the model as inconsistent, outputs carry `COVARIANCE_UNRELIABLE` and are not `valid`.
With that rule every *valid* outage sample is inside the 99.7 % region in the blackout, saturation, biased-IMU
and vibration scenarios, at the cost of declaring 96 % of those outage samples invalid. The dark-room case
(camera degraded but still accepted) stays at 70–76 % with and without the network: on this synthetic camera
model, low-health fixes are biased beyond the calibrated trust table. Calibrating the network's covariance
remains open (§7)."""),
]
for a, b in reps:
    assert a in s, a[:50]
    s = s.replace(a, b)
d.write_text(s, encoding="utf-8", newline="")
print("ok")
