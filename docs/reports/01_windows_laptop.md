# Report 1: Windows laptop (Intel i7-11800H, WSL) — merged `main` at c382494, 16 Nov 2026

The first independent benchmark of the merged repository, run on a Windows laptop (Linux under WSL) from the frozen
bundle `20261116-main-c382494-benchmark`, with no model tuning. Original:

![Windows laptop benchmark report](windows_laptop_c382494.webp)

## Numbers (transcribed)

| | Value | Limit |
|---|---:|---:|
| C++ entry, worst CPU | **13.97 ms** per second of IMU | 20 |
| C++ entry, peak memory | **5.06 MiB** | 8 |
| Output contracts (rows, timestamps, finite poses, unit quaternions) | **114 / 114** | |

Camera + IMU trajectory benchmark (the real-vision work, #26), TUM-VI development recordings, 38 scenarios, mean
aligned ATE (m):

| | Path Finder | Physics only | Standing still | Path Finder, clean |
|---|---:|---:|---:|---:|
| room1 | **0.625** | 6.135 | 1.010 | 0.456 |
| room2 | **0.893** | 2.476 | 1.244 | 0.358 |

Path Finder beats both baselines on overall ATE in **38/38** scenarios. Caveat recorded by the report: during the
blind period, room2 mean error is 0.701 m vs 0.662 m standing still.

Tests: C++ stress 45 passed, 2 skipped; root suite (Windows) 250 passed, 5 skipped, 2 failed initially; demo line
85 passed, 26 skipped, 1 xfailed, 1 failed.

## What the two failures are

- **Python runtime 31.3 ms/s vs 20.** The Python file is the readable reference, not the submission; the C++ entry is
  what is scored. On this Mac the same Python code takes 5.7 ms/s; the laptop measured under WSL with other work running.
- **1,000 Hz demo at 0.61× real time.** A stress test of the demo-line streaming tracker at five times the normal
  IMU rate; not scored.

## What changed after this report

Measured on `c382494`. Since then (#30, #31): input layout detection for the organisers' format, the convolution
reads contiguous memory and the entry builds with `-O3` (output unchanged bit for bit). The resource numbers here
are the slowest machine we have measured; see the comparison in [README.md](README.md).
