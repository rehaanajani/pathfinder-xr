# Timing checks

Real-time numbers taken while other tests or agents share the machine are
exploratory. They are not a pass or a fail of the 1× requirement.

## What owns the demo 1 kHz check

The assertion is `demo_line/tests/stress/test_nasty_inputs.py::test_other_imu_rates`
at 1,000 Hz (`rt_factor > 1`). It stays in the suite. Do not relax it.

Acceptance evidence is one of:

- `cd demo_line` then `python tools/run_isolated_timing.py`
- the `demo-timing` job in `.github/workflows/demo_line.yml` (`timing`), which runs
  that script alone on its own runner

The script sets one BLAS thread, starts no other benchmark, and writes a unique
`results/runs/<UTC>-demo-timing-<id>/timing.json` (`pathfinder.timing-run/1`).
An explicit `--out` must not already exist. Measurement failures fail the command
even if the subsequent pytest invocation succeeds.
It cannot see unrelated host load. Run it when the machine is otherwise idle.
The parent re-runs this after parallel work stops. The Python entry 20 ms/s check
is not this script; that measurement belongs to the entry owner and the same
isolation rule.

`tools/check_repo.py` compiles in memory and does not write `__pycache__`, so two
repository checks can run at once without colliding on bytecode files.
