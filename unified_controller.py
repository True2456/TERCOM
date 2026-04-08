import time
import math
import cv2
import numpy as np
from telemetry.telemetry_bridge import TelemetryBridge
from simulator.map_renderer import MapRenderer
from navigation.matcher import FFTMatcher
from utils.visualizer import PathVisualizer
import rasterio
from pyproj import Transformer

# --- CONFIGURATION ---
TARGET_ALT = 50.0  # Transition altitude
TRN_THRESHOLD = 15.0 # Mission-Grade PSR Threshold
DEM_PATH = "5m_DEM.tif"
WP_RADIUS = 30.0
TRN_SCALE = 0.15 # Validated DJI Scale
TRN_SEARCH_RADIUS = 300 # Localized 300m search

def main():
    print("[UNIFIED] Initializing Tactical Mission Controller...")
    # Connect on 14551 (Proven working port)
    commander = TelemetryBridge(connection_string='udpout:127.0.0.1:14551', source_system=255, wait_for_heartbeat=False)
    commander.start()

    # Orbit points around the headland (300m radius)
    waypoints = [
        (-29.985, 153.228), # WP1: North
        (-29.987, 153.231), # WP2: East (Cliffside)
        (-29.990, 153.228), # WP3: South
        (-29.987, 153.225), # WP4: West
    ]
    waypoint_idx = 0
    state = "INIT"
    gps_is_killed = False
    
    viz = PathVisualizer(waypoints)

    print("[TRN] Initializing Map and Matcher Engine...")
    renderer = MapRenderer(DEM_PATH)
    matcher = FFTMatcher(target_size=(800, 800))
    
    # Coordinate Transformer for MGA to Lat/Lon feedback
    to_wgs84 = Transformer.from_crs(renderer.crs, "EPSG:4326", always_xy=True)

    try:
        while True:
            pose = commander.get_pose()
            
            # --- MISSION STATE MACHINE ---
            if state == "INIT":
                if pose['lat'] != 0:
                    print(f"[MISSION] SITL Connected. Pose: {pose['lat']:.6f}, {pose['lon']:.6f}")
                    state = "ARMING"
            
            elif state == "ARMING":
                print("[MISSION] Setting GUIDED and ARMING...")
                commander.set_mode("GUIDED")
                time.sleep(1)
                commander.arm()
                time.sleep(2)
                if pose['armed']:
                    print(f"[MISSION] Armed. Taking off to {TARGET_ALT}m...")
                    commander.takeoff(TARGET_ALT)
                    state = "TAKEOFF"

            elif state == "TAKEOFF":
                if pose['alt_agl'] > 5.0:
                    print("[MISSION] Altitude threshold cleared. Engaging TRN Loop.")
                    state = "TRANSIT"
            
            elif state == "TRANSIT":
                # Check waypoint arrival
                target = waypoints[waypoint_idx]
                dist = math.sqrt((pose['lat'] - target[0])**2 + (pose['lon'] - target[1])**2) * 111111
                
                if dist < WP_RADIUS:
                    print(f"[MISSION] Reached Waypoint {waypoint_idx}. Advancing...")
                    waypoint_idx = (waypoint_idx + 1) % len(waypoints)
                    
                    # Simulate GPS Failure at Waypoint 2
                    if waypoint_idx == 2 and not gps_is_killed:
                        print("[MISSION] !!! SIMULATING PRIMARY GPS FAILURE !!!")
                        commander.mav.mav.command_long_send(
                            commander.mav.target_system, commander.mav.target_component,
                            31010, 0, 1, 0, 0, 0, 0, 0, 0
                        )
                        gps_is_killed = True

                # Send Position Command
                commander.set_position(target[0], target[1], TARGET_ALT)

            # --- REAL-TIME TRN MATCHING LOOP ---
            if pose['alt_agl'] > 10.0:
                # 1. Capture 'Virtual Drone View' from Simulator
                live_frame = renderer.render(pose['lat'], pose['lon'], pose['alt_agl'])
                
                if live_frame is not None:
                    # 2. Get Search Tile from DEM around LKP
                    e_lkp, n_lkp = renderer.latlon_to_mga(pose['lat'], pose['lon'])
                    
                    with rasterio.open(DEM_PATH) as src:
                        # Translate world MGA to local pixel window
                        px_c, px_r = ~src.transform * (e_lkp, n_lkp)
                        win = rasterio.windows.Window(int(px_c - 200), int(px_r - 200), 400, 400)
                        tile_dem = src.read(1, window=win)
                        
                        # Hillshade current terrain map
                        tile_hs = renderer._calculate_hillshade(*renderer._compute_gradients(tile_dem), 0, 45)
                        tile_hs = cv2.resize((tile_hs * 255).astype(np.uint8), (800, 800))

                    # 3. Perform FFT Edge-Match (Validated Parameters)
                    # Prepare Live Frame (Scale & Padding)
                    h, w = live_frame.shape; w_s = int(800 * TRN_SCALE); h_s = int(w_s * (h/w))
                    resized = cv2.resize(live_frame, (w_s, h_s))
                    padded = np.zeros((800, 800), dtype=np.uint8)
                    y_o, x_o = (800-h_s)//2, (800-w_s)//2
                    padded[y_o:y_o+h_s, x_o:x_o+w_s] = resized
                    
                    dx, dy, psr = matcher.match(padded, tile_hs, denoise=True, edge_match=True)
                    
                    if psr > TRN_THRESHOLD:
                        # 4. Resolve Correct Positioning
                        # Convert pixel shift to global row/col
                        global_r = px_r - 200 + (y_o + dy + h_s/2)/2.0
                        global_c = px_c - 200 + (x_o + dx + w_s/2)/2.0
                        
                        trn_e, trn_n = src.transform * (global_c, global_r)
                        trn_lon, trn_lat = to_wgs84.transform(trn_e, trn_n)
                        
                        # Submit Tactical Fix
                        commander.send_gps_input(trn_lat, trn_lon, pose['alt_agl'])
                        match_str = f"LOCK (PSR: {psr:.1f})"
                    else:
                        match_str = f"REJECTED (PSR: {psr:.1f})"
                else:
                    match_str = "MAP_BOUNDARY"
            else:
                match_str = "WAITING_ALT"

            # Update Visualizer
            viz.update(pose['lat'], pose['lon'], state, match_str)

            # Dashboard
            stat_str = "[FAIL]" if gps_is_killed else "[GPS1]"
            print(f"[UNIFIED] {stat_str} | State: {state} | Alt: {pose['alt_agl']:.1f}m | WP: {waypoint_idx} | TRN: {match_str}")
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("[UNIFIED] Shutting down...")
        commander.stop()

if __name__ == "__main__":
    main()
