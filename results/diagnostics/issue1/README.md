# Issue #1 probes and evidence

All data here is **synthetic** (`pathfinder.synthetic.head_walk`, simulated camera fixes, simulated co-located
redundant IMUs). Nothing in this folder is hardware evidence or a TLIO / TUM-VI benchmark result.
Run every script from the repository root. Scripts marked *baseline* target the `ff7ec44` API and were run in a
detached worktree of that commit; *both* runs on either version.

| Script | Runs on | What |
|---|---|---|
| `p1.py` | baseline | audit probe: scalar confidence, clean parity false alarms, lever-arm false faults |
| `p4.py` | baseline | audit probe: wrong fix under `FilterConfig()` vs `ours_cfg()` |
| `rc1_wrongfix_rootcause.py` | baseline | instrumented root cause of the wrong-fix failure (q_scale, NIS, gate) |
| `wrongfix_matrix.py` | both | wrong-fix / drift / jump / repeated / blackout matrix, shipped config, seeds 0-3 |
| `coverage_probe.py` | both | position-uncertainty coverage (scalar before, chi2(3) after) |
| `p1_after.py` | new | findings 2-4 re-measured with the new monitor |
| `synthetic_eval_after.py` | new | the audit's synthetic scenario table, without writing `results/` |
| `model_formulation.py` | new | learned-update formulations with the models-v1 network on synthetic walks |
| `trace_episode.py`, `trace_model_wrongfix.py`, `smoke_refree.py` | new | single-case traces used while debugging |
| `fill_docs.py`, `fix_doc_paths.py`, `edit_readmes.py` | new | generated the tables / wording edits in `docs/ESTIMATOR.md` and the READMEs |

`evidence/` (text converted to UTF-8):

| File | Content |
|---|---|
| `*_baseline.txt`, `audit_synthetic_eval_ff7ec44.log` | before (`ff7ec44`) |
| `wrongfix_matrix_attempt1_regression.txt` ... `attempt12_*.txt` | every intermediate attempt, kept including failures |
| `wrongfix_matrix_final.txt` | final code |
| `synthetic_eval_attempt_no_veto.*`, `synthetic_eval_attempt_veto_all_episodes.*` | failed attempts (4.8 m wrong-fix ATE; 1276 m saturation divergence) |
| `*_before_distrust_outage_rule.*` | results just before the last rule (identical positions; validity changed) |
| `synthetic_eval_after.*` | final code |
| `coverage_attempt_*.txt`, `coverage_new.txt` | uncertainty coverage, intermediate and final |
| `imu_fdi_evaluation*.log` | IMU monitor statistics. `imu_fdi_evaluation.log` used a wrong p-value transform for the KS check (superseded). `..._final_head.log` is a run cut after two lines by a shell pipe; `..._final.log` is the complete run behind `../imu_fdi_evaluation.json` |
| `pytest_full.txt` | full test-suite result at the time of writing |
| `issue1_body_2026-11-14T17-04Z.md` | issue text as read |

Models: `models/dispnet_faults/best.pt`, SHA-256 `611E97FD...C3EBA3`, identical to the `models-v1` release
asset `dispnet_faults.pt`.
