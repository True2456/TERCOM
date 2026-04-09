# TERCOM: Optical Terrain Referenced Navigation (TRN)

<img width="800" height="800" alt="drone_path_viz" src="https://github.com/user-attachments/assets/cec2f26d-930f-42ca-be4d-1708bc410885" />

This repository contains a prototype software stack for simulating **Optical Terrain Referenced Navigation (TRN)** tailored for autonomous VTOL drones operating in GNSS-denied environments. 

It implements a purely vision-based localization pipeline that compares downward-facing camera imagery against known georeferenced satellite/orthophoto maps in real-time, feeding accurate positional data back to the flight controller when the primary GPS fails.

## Key Features

- **Structural Template Matching**: Uses Canny Edge extraction and Normalized Cross-Correlation (`cv2.matchTemplate`) to match live camera frames to maps. This method ignores transient shadows/colors and focuses entirely on invariant physical topology (buildings, roads).
- **EKF Kinematic Gate**: An Extended Kalman Filter-style gate that protects the flight controller. It compares the implied optical speed between matches against the drone's IMU speed. If an optical match generates an impossible speed jump, it is rejected.
- **Python VTOL SITL**: A lightweight, fast MAVLink-based Software-In-The-Loop (SITL) simulator simulating VTOL flight dynamics to test TRN architecture without risk to physical airframes.
- **Flight Visualization**: High-resolution tracking maps that cleanly map out GPS truth, accepted optical locks, and EKF rejections for post-flight analysis.

---

## Architecture Overview

The system operates across three core nodes communicating via asynchronous UDP MAVLink:

1. **`simulate_ekf.py` / `unified_controller.py`**: The cognitive core. It calculates camera footprint, extracts map patches, executes OpenCV structural matching, filters outliers through the EKF, and injects synthetic `GPS_INPUT` messages back to the MAVLink router.
2. **`simulator/vtol_sim.py`**: Simulates the drone physics, processing waypoints, generating IMU/attitude telemetry, and listening for TRN-derived GPS injections.
3. **`scripts/live_map_viz.py`**: A live OpenCV dashboard that intercepts telemetry to draw the drone's active footprint and TRN vectors over the orthophoto in real time.

---

## Setup & Dependencies

It is highly recommended to run this in a Python Virtual Environment (`venv`).

```bash
pip install numpy opencv-python rasterio pyproj pymavlink pillow
```

You must have a highly accurate Orthophoto/GeoTIFF Map caching the flight area (e.g., `RedRock Pi color 1 res.tif`) natively indexed in a projected coordinate system (e.g., EPSG:28356).

---

## 🚀 Running the Stack

### 1. Live SITL Simulation
To launch the full Drone Physics Simulator -> Unified TRN Controller -> Live Map Visualizer stack:

```powershell
.\start_trn_stack.ps1
```
The drone will automatically take off to the `TARGET_ALT`, transition to horizontal flight, and begin navigating its Waypoints. The controller will log all match parameters out to `sitl_flight_history.json`.

**Post-Flight Map Generation:**
After successfully concluding a SITL flight, generate the high-resolution flight track graphing your optical locks:
```powershell
python scripts\plot_sitl.py
```

### 2. Offline Optical Validation (`test_tm.py`)
If you have a folder of raw Nadir DJI photos (e.g. `Test photos 3/`) and the corresponding `gps_ground_truth.json` database, you can run the offline test suite to graph the raw accuracy of the Template Matcher without activating the physics engine.

```powershell
python scripts\test_tm.py
```
This will compute the error on every frame individually and output an annotated dataset map.

---

## Future Roadmap: Video Integration
The stack is currently architected around static waypoint matching. Moving forward, integrating continuous high-framerate `.MP4` video coupled with synchronized `.SRT` telemetry logs will unlock:
1. **Optical Flow (Lucas-Kanade)**: Dramatically shrinking the search window size to speed up the loop from ~2 Hz to >10 Hz.
2. **Temporal Consistency**: Using previous positive locks to permanently discard geographic ghost-matches.
3. **Micro-SBC Optimization**: Pivoting from dense map correlation to Sparse Feature Matching + C++ implementations for Raspberry Pi / Edge node deployments.
