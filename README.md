# TERCOM: Optical Terrain Referenced Navigation (TRN)

<img width="600" height="600" alt="sitl_ekf_simulation_map" src="https://github.com/user-attachments/assets/efdeddd1-4e5e-4382-8fb1-616a01f8df5a" />

This repository contains a prototype software stack for simulating **Optical Terrain Referenced Navigation (TRN)** tailored for autonomous VTOL drones operating in GNSS-denied environments. 

It implements a purely vision-based localization pipeline that compares downward-facing camera imagery against known georeferenced satellite/orthophoto maps in real-time, feeding accurate positional data back to the flight controller when the primary GPS fails.

## Key Features

- **Hybrid Video Tracking**: Combines 30Hz Lucas-Kanade (LK) Optical Flow with 0.5Hz Structural Anchor locking. This provides smooth, low-drift positioning with absolute geographic corrections.
- **Heading-Aware Projection**: Correctly projects camera-frame pixel shifts to world Easting/Northing offsets using drone heading and altitude (AGL).
- **Best-Patch Extraction**: Automatically identifies the most structurally-rich (high edge density) regions of a video frame as templates, ensuring reliable locks even over featureless terrain.
- **Flight Visualization**: High-resolution tracking maps with error gradients, smoothed GPS traces, and comprehensive per-leg performance statistics.

---

## Architecture Overview

The system operates across three core nodes and an offline processing pipeline:

1. **`simulate_ekf.py`**: Live SITL controller for waypoint missions.
2. **`scripts/process_video.py`**: High-speed offline processor for airborne `.MP4` video and `.SRT` telemetry.
3. **`scripts/plot_sitl.py`**: Generates detailed analytical maps from flight history JSON.
4. **`simulator/vtol_sim.py`**: MAVLink-based physics engine for SITL testing.

---

## Setup & Dependencies

It is highly recommended to run this in a Python Virtual Environment (`venv`).

```bash
pip install numpy opencv-python rasterio pyproj pymavlink pillow
```

You must have a georeferenced Orthophoto/GeoTIFF Map caching the flight area (e.g., `RedRock Pi color 1 res.tif`) natively indexed in a projected coordinate system (e.g., EPSG:28356).

---

## 🚀 Running the Stack

### 1. Video-Based TRN Processing
To process a drone video file (`.MP4`) with a synchronized subtitle (`.SRT`) log:

1. Configure `VID_PATH`, `SRT_PATH`, and `ORTHO_PATH` in `scripts/process_video.py`.
2. Run the tracker:
   ```powershell
   python scripts\process_video.py
   ```
3. Generate the visualization map:
   ```powershell
   python scripts\plot_sitl.py
   ```

### 2. Live SITL Simulation
To launch the full Drone Physics Simulator -> Unified TRN Controller -> Live Map Visualizer stack:

```powershell
.\start_trn_stack.ps1
```
The drone will automatically take off, transition to horizontal flight, and navigate its waypoints. The controller logs all match parameters to `sitl_flight_history.json`.

---

## Future Roadmap

1. **Hardware Acceleration**: Migrating the LK flow and template matching loops to C++ (using OpenCV `UMat` or OpenCL) for real-time performance on Raspberry Pi / Edge nodes.
2. **Multi-Spectral Support**: Incorporating IR/Thermal camera support for night-time TRN operations.
3. **Terrain Correlation (TERCOM)**: Adding 3D elevation profile matching (using DEM data) to augment 2D visual feature tracking.
