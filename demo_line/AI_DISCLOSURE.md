# AI disclosure

**Tools used:** Claude (Anthropic) through Claude Code, as a coding assistant. Gemini and Grok were used for literature and prior-work search.

**What AI helped with**
- Literature search: finding prior work (TLIO, RoNIN, EqNIO, AirIMU, AirIO, KISS-IMU, Samsung's DVI-SLAM) and the TLIO headset dataset.
- Writing and refactoring code: the fault injector, the Kalman filter, the IMU voting logic, the training script, the benchmark and the report.
- Drafting documentation, which we then edited.

**What we decided and checked ourselves**
- The problem framing, the choice of approach (per-sensor fault isolation + camera health + a fault-aware learned motion model) and the scenarios to test.
- Every number in the README comes from `evaluate.py` running on held-out data. Nothing is hand-entered.
- Every source we cite was opened and read. Claims we could not verify are not in the README.

**What no AI did**
- No AI model runs inside the system. The tracker is a Kalman filter plus a small network that we trained ourselves on public data.
- No result was generated or estimated by a language model.
