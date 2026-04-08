import os
import sys
sys.path.append(os.getcwd())
import time
import math
from telemetry.telemetry_bridge import TelemetryBridge

# --- MISSION CONFIG ---
RED_ROCK = (-29.9858, 153.2308) # Headland Coordinate 🏔️
TARGET_ALT = 80.0
CRUISE_SPEED = 25.0
GPS_KILL_THRESHOLD = 0.005 # Lat/Lon degrees to headland (~500m)

def get_orbit_points(center, radius_m, points=8):
    lat, lon = center
    coords = []
    for i in range(points):
        angle = math.radians(i * (360 / points))
        # Approx conversion (1m = 1/111111 degrees)
        dlat = (radius_m * math.cos(angle)) / 111111.0
        dlon = (radius_m * math.sin(angle)) / (111111.0 * math.cos(math.radians(lat)))
        coords.append((lat + dlat, lon + dlon))
    return coords

def main():
    print(f"[MISSION] Initializing Red Rock Orbit Challenge... (SITL Connected)")
    commander = TelemetryBridge(connection_string='udpout:127.0.0.1:14550', source_system=255, wait_for_heartbeat=False)
    commander.start()
    
    # Orbit points around the headland (300m radius)
    ORBIT_POINTS = get_orbit_points(RED_ROCK, 350.0)
    
    state = "INIT"
    waypoint_idx = 0
    gps_is_killed = False
    
    try:
        while True:
            pose = commander.get_pose()
            if not pose:
                time.sleep(0.1)
                continue
            
            # --- MISSION STATE MACHINE ---
            if state == "INIT":
                print(f"[MISSION] Current State: {state} | Armed: {pose['armed']} | Mode: {commander.mav.flightmode}")
                if not pose['armed']:
                    print("[MISSION] Attempting to ARM and set GUIDED...")
                    commander.set_mode("GUIDED")
                    time.sleep(1)
                    commander.arm()
                    # FORCE transition after small delay to overcome telemetry lag
                    time.sleep(2)
                    pose = commander.get_pose()
                    if pose['armed']:
                        print("[MISSION] Arming confirmed. Triggering TAKEOFF...")
                        commander.takeoff(TARGET_ALT)
                        state = "TAKEOFF"
                    else:
                        print("[MISSION] Arming not yet confirmed by telemetry. Retrying...")
                else:
                    print(f"[MISSION] Already ARMED. Triggering TAKEOFF to {TARGET_ALT}m...")
                    commander.takeoff(TARGET_ALT)
                    state = "TAKEOFF"
            
            elif state == "TAKEOFF":
                if pose['alt_agl'] >= TARGET_ALT * 0.9:
                    print(f"[MISSION] Altitude reached. Transiting to Red Rock Headland...")
                    state = "TRANSIT"
            
            elif state == "TRANSIT":
                dist = math.sqrt((pose['lat'] - RED_ROCK[0])**2 + (pose['lon'] - RED_ROCK[1])**2)
                
                # GPS KILL TRIGGER: Halfway there / Near the headland
                if not gps_is_killed and dist < GPS_KILL_THRESHOLD:
                    print("\n" + "!"*40)
                    print("[MISSION] CRITICAL: GPS1 FAILURE SIMULATED! (OFFLINE)")
                    print("[MISSION] TRN SYSTEM TAKING PRIMARY CONTROL...")
                    print("!"*40 + "\n")
                    # Disable GPS1 in simulator (1=GPS1, 2=GPS2)
                    commander.mav.mav.command_long_send(1, 1, 31010, 0, 0, 1, 0, 0, 0, 0, 0)
                    gps_is_killed = True
                
                commander.goto(RED_ROCK[0], RED_ROCK[1], TARGET_ALT)
                
                if dist < 0.001: # Arrived at headland center (approx 100m)
                    print(f"[MISSION] Headland reached. Starting orbit patterns...")
                    state = "ORBIT"
            
            elif state == "ORBIT":
                target_lat, target_lon = ORBIT_POINTS[waypoint_idx % len(ORBIT_POINTS)]
                dist_to_wp = math.sqrt((pose['lat'] - target_lat)**2 + (pose['lon'] - target_lon)**2)
                
                # Move to next orbit point
                if dist_to_wp < 0.0005: 
                    waypoint_idx += 1
                    print(f"[MISSION] Orbit Point {waypoint_idx % len(ORBIT_POINTS)} reached. Sweeping...")

                commander.goto(target_lat, target_lon, TARGET_ALT)

            # Dashboard (Truncated for brevity)
            p_str = "[FAIL]" if gps_is_killed else "[GPS1]"
            print(f"[RED_ROCK] {p_str} | Mode: {state} | Alt: {pose['alt_agl']:.1f}m | WP: {waypoint_idx}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("[MISSION] Aborted by user.")
    finally:
        commander.stop()

if __name__ == "__main__":
    main()
