# Requirements tracking (issue #3 A)

## Sources found in the repository (2026-11-15)

| Source | Content | Status |
|---|---|---|
| `README.md` header | "Samsung PRISM XR Hackathon 2026, Track 3: Path Finder, Trajectory Estimation Under Sensor Degradation", team name, institution, and a one-sentence problem quote | Written by the team. The organizer document behind it is **not in the repository** |
| `README.md` "Submission package" | Deck, video, demo, results, models, AI disclosure | The team's own list. It is not shown to be the organizer's required list |
| `docs/industry_research.md`, `docs/research_raw/` | Public examples and background | Assumptions and examples only. They do not state requirements |

No official challenge statement, deadline or time zone, submission link, rounds, eligibility rules,
judging rubric, hardware or data restrictions, or artifact format was found. This change did not
contact organizers, and it does not guess these facts.

## Verified vs assumed

| Topic | Verified | Assumed / unknown | Owner action |
|---|---|---|---|
| Problem statement | One sentence (quoted in the README) | Full brief, scoring | Obtain the official brief |
| Deadline and time zone | – | Unknown | Confirm with organizers |
| Required artifacts | – | Deck, video, demo, code (the team's list) | Confirm the list and formats |
| Hardware/device requirement (APK, headset, latency limit) | – | Do **not** infer from other teams' repositories | Confirm |
| Data restrictions / licences | TLIO is CC BY-NC 4.0; TUM-VI is CC BY 4.0 | Whether non-commercial training data is acceptable | Organizers and owners |
| Intended deliverable | Research library + offline replay (this repository) | Whether a runtime component is expected | Owners |

## Requirement → evidence map (current)

| Requirement (team-derived) | Implementation | Test | Result | Artifact | State |
|---|---|---|---|---|---|
| Track under degraded camera | `stereo_vo.py` + relative updates | `test_relative_pose.py`, `test_smoke_pipeline.py` | `results/vision/` | `demo/vision_replay.html` | Implemented and evaluated on one development recording |
| Track under degraded IMU | `imu_health.py`, IMU fault scenarios | issue #1 tests | `results/vision/` (single real IMU) | – | Single real IMU only. Multi-IMU results are simulated |
| Honest uncertainty / validity | reason codes, `COVARIANCE_UNRELIABLE` | runtime safety tests | NEES coverage in the report | – | Covariance not validated on held-out data |
| Real-time feasibility | `tools/profile_pipeline.py` | – | Laptop profile in the report | – | Laptop only. No device measurement |
| Reproducibility | lock, CI, `fetch_models.py` | `test_release_tools.py` | Fresh-venv check | `docs/REPRODUCE.md` | Second-person reproduction open |
| Submission package | – | – | – | Deck and video not regenerated | **Blocked** on organizer requirements and owner decisions |

Owners and reviewers are not assigned in the repository. That is an issue #3 A action for the team.
