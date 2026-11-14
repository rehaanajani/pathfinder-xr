# Grok answer: who builds headset tracking, with what (14 Nov 2026)

Mode: Fast (Expert, Auto and Heavy need a paid plan). Grok reported 121 sources but its page showed no per-claim links, so treat specifics as leads to verify.

**Prompt:** head tracking tech of Meta, Apple, Google + Samsung, Qualcomm, Pico, Snap, XREAL, Varjo, Magic Leap; startups; failure reports; open-source repos; head-to-head comparisons.

## Company tracking tech
- **Meta Quest 3 / 3S:** inside-out visual-inertial SLAM with multiple outward cameras + IMU.
- **Meta Orion (prototype):** on-glasses SLAM/VIO at high rate (about 90 Hz cited in talks), 4-7 outward cameras plus inward eye cameras, custom low-power silicon bringing SLAM/hand/eye tracking to tens of milliwatts; a wireless puck does heavier compute.
- **Meta Aria Gen 2:** 6DoF VIO with 4 CV cameras (more stereo overlap), IMU, on-device perception coprocessor; offline MPS gives denser SLAM maps.
- **Meta Ray-Ban Display:** no positional tracking or world-locking; the EMG Neural Band is input, not head tracking.
- **Apple Vision Pro:** inside-out VIO with 12-14 cameras (outward, inward eye tracking) plus LiDAR/depth and multiple IMUs; LiDAR helps in low light.
- **Samsung Galaxy XR (Android XR, XR2+ Gen 2):** 6DoF inside-out with 6 outer tracking cameras, 2 passthrough cameras, a dToF depth sensor, multiple IMUs, 4 eye-tracking cameras, hand tracking.
- **Qualcomm XR2+ Gen 2:** 12+ concurrent cameras, dedicated XR / computer-vision hardware blocks for low-power 6DoF head tracking; Snapdragon Spaces is the SDK layer.
- **Pico (4 Ultra):** camera + IMU SLAM/VIO on XR2 Gen 2, DSP-accelerated vision, improved large-space and moving-vehicle modes; Motion Trackers mix IMU + IR LEDs.
- **Snap Spectacles:** "Snap Spatial Engine": 2 RGB + 2 IR CV cameras + 6-axis IMUs; one processor runs 6DoF SLAM, hands and scene understanding at about 80 mW for the tracker; IMU-driven poses at about 800 Hz.
- **XREAL:** mostly 3DoF; the XREAL Eye add-on gives monocular 6DoF on the X1 chip; Air 2 Ultra has stereo cameras + IMU; weak in low light and outdoors; community Basalt ports for Linux/OpenXR.
- **Varjo:** high-end passthrough; SteamVR outside-in option on XR-4; LiDAR/RGB depth on XR-3; Polhemus electromagnetic tracking for vehicles where optical fails.
- **Magic Leap 2:** inside-out cameras + dual IMUs + CVIP engine.

## Startups
- **Spectacular AI** (Helsinki, about 2021): VISLAM SDK on embedded CPUs; now drone/robot GPS-free positioning focus.
- **Lynx:** open-sourced an ORB-SLAM3 Android port tuned for Qualcomm Hexagon DSP, about 90 fps on XR2 Gen 1 at 640x400.
- **SLAMcore** (London): stereo-camera SLAM for industrial vehicles (forklifts); 30+ facilities; about $40M raised including a $14M 2026 round (Rockwell Automation, Toyota Ventures); earlier pilots included Meta.
- Others: LP-Research (sensor-fusion demos with Quest).

## Failure reports and how products cope
- Common failures: dark rooms, fast head turns, blank walls, IMU drift once vision drops.
- Quest: low-light tracking loss and boundary shifts; users add IR floodlights; some OS updates broke IR illuminators.
- Vision Pro: LiDAR keeps head/hand tracking working even in pitch black in user tests.
- Mitigations everywhere: IMU bridging short gaps, multi-camera redundancy, IR illumination, depth/LiDAR, map persistence.

## Open-source repos
- openxrlab/xrslam (~599 stars): VIO toolbox and benchmark, desktop + mobile.
- Lynx ORB-SLAM3 Android port (2025).
- DeskUnreal/xreal-vio-vr (~16 stars): Basalt + Monado pose injection for XREAL.
- 93won/openvio / lightweight_vio (~395 stars): MIT VIO, mono/stereo, online calibration.
- Monado integrations: Basalt, ORB-SLAM3, Kimera-VIO; Monado SLAM dataset for XR headsets.
- FoundationVIO (ori-drs, ~97 stars): foundation-model + IMU VIO.

## Head-to-head comparisons
- *Apple vs. Meta: A Comparative Study on Spatial Tracking in SOTA XR Headsets* (2024): Vision Pro beat Quest 3 by 33.9 % RPE and 14.6 % APE; both over 10 cm APE in hard conditions.
- *XR Reality Check: What Commercial Devices Deliver for Spatial Tracking* (arXiv 2025): Vision Pro, Quest 3, HoloLens 2, Magic Leap 2, XREAL Air 2 Ultra plus an ORB-SLAM3 baseline. Vision Pro best (RPE about 0.52 cm); others 1.5-3x worse, more in hard conditions. *Verify numbers before citing.*
