"""Apply identical narrative corrections to README.md and README_template.md (numbers untouched)."""
from pathlib import Path

REPL = [
("""2. **Parity-space IMU fault detection.** All IMUs on a headset sit on one rigid frame, so their readings must
   agree. The aerospace parity test (Sturza 1988) projects out the true motion, flags a fault with a chi-square
   test and isolates the faulty IMU. With 3+ IMUs it catches small biases and slow drift that simple voting
   misses. Galaxy XR has five IMUs.""",
"""2. **Parity-space IMU fault detection.** IMUs on one rigid frame must agree once their readings are expressed at
   the same point. The aerospace parity test (Sturza 1988) projects out the true motion; its noise statistics are
   learned causally during a warm-up (Hotelling prediction threshold, outlier-trimmed, stationarity-checked), and a
   faulty IMU is isolated with 3+ sensors. Agreement is *relative*: an error common to every IMU is invisible to it.
   Supported today: co-located, aligned, time-synchronised sensors (what the simulated redundant IMUs are). A
   lever-arm / mounting-rotation model exists but is validated only on synthetic rigid-body data, not hardware.""",),
("""4. **An adaptive error-state Kalman filter** blends everything by trust. A soft chi-square gate caps any reading
   that disagrees, a camera that suddenly jumps is held out for up to 10 s, and after an outage the filter
   re-anchors within 1 s. When healthy camera frames keep disagreeing with the IMU, the IMU's process noise is
   raised (adaptive Kalman filtering), so drifting IMUs are corrected faster.""",
"""4. **An adaptive error-state Kalman filter** blends everything by trust. Camera fixes pass a hard chi-square
   gate. A camera that disagrees after sustained agreement is treated as a suspect jump and held out for at least
   10 s; after an outage the filter re-anchors after 1 s of mutually consistent fixes (all durations in seconds,
   see `docs/ESTIMATOR.md`). Adaptive process noise is driven only by fixes that passed the gate, so a rejected fix
   cannot raise its own influence.""",),
("""6. **A health signal out.** Per-sensor trust and the filter's own position uncertainty in metres. OpenXR today
   only reports a yes/no "position tracked" flag; this gives a graded one.""",
"""6. **A health signal out.** A tracking state, validity flag and reason codes, the IMU monitor state, and the
   filter's 3x3 position covariance (evaluate it as a chi-square(3) region, not as a scalar 3-sigma radius).
   OpenXR today only reports a yes/no "position tracked" flag. The runtime interface is the reference-free
   `pathfinder.estimator.StreamingEstimator`; benchmark runs start from the reference pose and are labelled so.""",),
("""## Results
""",
"""## Results

> **These tables were produced by the estimator at `ff7ec44` (before the issue #1 correctness work) and have not
> been regenerated.** The IMU monitor, camera supervision and learned-model rules changed since, so the numbers
> describe the earlier system. Re-running the TLIO/TUM-VI benchmarks needs the datasets and is tracked in issues
> #2 and #3; synthetic before/after checks are in `docs/ESTIMATOR.md`.
""",),
("""- **Extra IMUs are simulated.** TLIO and TUM-VI have one IMU each. Multi-IMU tests copy the real IMU with its own
  noise and bias per copy. The parity-space detector models lever arms but is not tuned on real multi-IMU hardware.""",
"""- **Extra IMUs are simulated.** TLIO and TUM-VI have one IMU each. Multi-IMU tests copy the real IMU with its own
  noise and bias per copy, co-located and aligned. Real multi-IMU support (measured mounting geometry, time
  offsets, synchronised raw streams) is unvalidated and must not be claimed until hardware data exists.""",),
("""- **Two IMUs detect, three isolate.** With two IMUs a smooth bias is detected but cannot always be attributed; the
  parity test isolates it with three or more, which is why Galaxy XR's five matter.""",
"""- **Two IMUs detect, three isolate.** With two IMUs a smooth bias is detected but cannot be attributed (the
  monitor reports it as unattributed and inflates process noise); three or more isolate one faulty sensor. Errors
  common to every IMU cannot be detected by comparing them.""",),
]

for name in ("README.md", "README_template.md"):
    p = Path(name)
    s = p.read_text(encoding="utf-8")
    for old, new in REPL:
        if s.count(old) != 1:
            raise SystemExit(f"{name}: expected one match for: {old[:60]!r}")
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="")
    print("updated", name)
