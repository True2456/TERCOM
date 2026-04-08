import os
# Force MAVLink 2.0 for ArduPilot SITL compatibility
os.environ['MAVLINK20'] = '1'

import time
import numpy as np
import cv2
from simulator.map_renderer import MapRenderer
from navigation.tile_manager import TileManager
from navigation.matcher import FFTMatcher
from telemetry.telemetry_bridge import TelemetryBridge
from utils.profiler import Profiler
import math

def main():
    # 1. Config
    DEM_PATH = '5m_DEM.tif'
    UPS_RATE = 2.0 # 2Hz as requested
    PSR_THRESHOLD = 15.0 # Confidence threshold for gating
    
    # 2. Components
    # Bridge to SITL First (Non-blocking)
    bridge = TelemetryBridge(wait_for_heartbeat=False)
    bridge.start()
    
    # Simulator (on PC RTX 3060) to generate "Live Feed" from SITL Pose
    renderer = MapRenderer(DEM_PATH)
    
    # Navigator (to be ported to Pi)
    tile_manager = TileManager(DEM_PATH)
    matcher = FFTMatcher()
    
    profiler = Profiler()
    
    print(f"TRN System Started at {UPS_RATE} Hz")
    
    current_lat = 0
    current_lon = 0
    initialized = False
    
    try:
        while True:
            start_t = time.time()
            
            # A. Get current SITL Pose
            pose = bridge.get_pose()
            if pose['lat'] == 0:
                # Wait for SITL to provide valid position and altitude
                # print("Waiting for valid pose from bridge...")
                time.sleep(0.5)
                continue
            
            if not initialized:
                current_lat = pose['lat']
                current_lon = pose['lon']
                initialized = True
                print(f"TRN Initialized at: {current_lat}, {current_lon}")
            
            if pose['alt_agl'] < 5.0:
                print(f"Waiting for takeoff... Currently at {pose['alt_agl']:.1f}m")
                time.sleep(1.0)
                continue
                
            # B. Simulate Live Camera Feed (using RTX 3060)
            with profiler.track("Simulate_Frame"):
                live_frame = renderer.render(
                    pose['lat'], pose['lon'], pose['alt_agl'],
                    pose['pitch'], pose['roll'], pose['yaw']
                )
            
            if live_frame is None:
                continue
                
            # C. Fetch Map Tile (Lightweight Pi Logic)
            with profiler.track("Fetch_Map_Tile"):
                # Simulating "last known" position from EKF or previous match
                # Use current pose but add some noise to simulate drift
                noisy_lat = pose['lat'] + np.random.normal(0, 0.0001)
                noisy_lon = pose['lon'] + np.random.normal(0, 0.0001)
                map_tile = tile_manager.get_tile_at(noisy_lat, noisy_lon, pose['alt_agl'])
            
            if map_tile is None:
                continue
            
            # D. Pre-process and Match
            with profiler.track("Match_Algorithm"):
                # Orthorectify live frame
                ortho_live = matcher.orthorectify(live_frame, pose['roll'], pose['pitch'], pose['alt_agl'])
                
                # FFT Phase Correlation
                dx_px, dy_px, psr = matcher.match(ortho_live, map_tile)
            
            # E. Gating Logic & Positioning
            if psr > PSR_THRESHOLD:
                # Convert pixel shift to meters (assuming 5m/px resolution)
                de_m = dx_px * 5.0
                dn_m = dy_px * 5.0
                
                # Apply correction to our estimate
                current_lat -= (dn_m / 111111.0)
                current_lon += (de_m / (111111.0 * math.cos(math.radians(current_lat))))
                
                # Injection to SITL (GPS_INPUT / Virtual GPS)
                bridge.send_gps_input(current_lat, current_lon, pose['alt_agl'])
                
                print(f"Match: OK (PSR: {psr:.1f}) | Offset: E={de_m:.1f}m, N={dn_m:.1f}m | Injecting: {current_lat:.6f}, {current_lon:.6f}")
            else:
                # Drift estimate
                current_lat += 0.000005
                current_lon += 0.000005
                print(f"Match: REJECTED (PSR: {psr:.1f}) < {PSR_THRESHOLD}")
                
            # F. Timing control
            elapsed = time.time() - start_t
            sleep_time = max(0, (1.0 / UPS_RATE) - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bridge.running = False
        tile_manager.close()

if __name__ == "__main__":
    main()
