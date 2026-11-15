# Requirements tracking (issue #3 A)

## Sources found in the repository (2026-11-15)

| Source | Content | Status |
|---|---|---|
| `README.md` header | "Samsung PRISM XR Hackathon 2026, Track 3: Path Finder, Trajectory Estimation Under Sensor Degradation", team name, institution, and a one-sentence problem quote | Written by the team. The organizer document behind it is **not in the repository** |
| `README.md` "Submission package" | Deck, video, demo, results, models, AI disclosure | The team's own list. It is not shown to be the organizer's required list |
| `docs/industry_research.md`, `docs/research_raw/` | Public examples and background | Assumptions and examples only. They do not state requirements |

**Update (same day, after fetching all branches):** branch `track3-entry` (PR #12, open, by the repository
owner) contains `docs/TRACK3_BRIEF.md`. That file restates the official deck
*XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf* (slides 14–18, 25–27). It names the coordinator who sent
the deck and states that the PDF is the authority. The PDF itself is **not** in the repository, so this change
treats the brief as a team restatement that is still to be verified against the PDF. According to it:

* **The scored Track 3 task is IMU-only.** Estimate a 6-DoF trajectory from a degraded EuRoC-format `imu0/data.csv`,
  write `estimated_trajectory.txt` (TUM format, one pose per IMU row), and dead-reckon 5 s after IMU failure.
  Budgets are ≤ 8 MB peak RAM and ≤ 20 ms per 1 s of IMU on a single thread. Scoring is 40 pts efficiency,
  30 checkpoints, 20 blind 5 s and 10 dropouts, measured with `evo` ATE/RPE.
* Deliverables: a zip with the trajectory, `src/main.py|cpp` and README; a public or shared repo with Docker; a
  video of 3 min or less; a deck named `SRMIST_CaliforniaBurrito`; one Google Form submission.
* Dates (tentative, and two sources conflict): final submission **18 Nov 2026 23:59** (deck) or 22 Nov (earlier
  email). The time zone is not stated.

**Consequence for this PR:** the real-vision work (issues #2/#3) is **not** the scored entry. It is innovation
and supporting evidence (the brief lists visual odometry as "not scored"). It does not touch the Track 3 entry
in PR #12. This PR does not submit anything, choose the deadline or change the entry.

## Verified vs assumed

| Topic | Verified | Assumed / unknown | Owner action |
|---|---|---|---|
| Problem statement | Team restatement of the official deck (`track3-entry:docs/TRACK3_BRIEF.md`) | Not checked against the PDF in this repository | Commit the PDF, or a verified excerpt, if redistribution is allowed |
| Deadline and time zone | – | 18 Nov or 22 Nov 2026 per the team brief (sources conflict; no time zone) | Confirm with organizers |
| Required artifacts | Per the brief: zip, `src/`, README + Docker, video of 3 min or less, deck `SRMIST_CaliforniaBurrito` | Exact form fields | Owners (PR #12) |
| Hardware/compute requirement | Per the brief: ≤ 8 MB RAM, ≤ 20 ms/s, single thread, offline (IMU entry) | No headset/APK requirement stated | – |
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
