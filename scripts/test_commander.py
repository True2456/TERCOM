import sys
import os
import time

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from telemetry.telemetry_bridge import TelemetryBridge

def main():
    # Connect to SITL as a GCS on dedicated Port 14552
    # Using 'udpin' to listen for SITL broadcast
    commander = TelemetryBridge(connection_string='udpin:0.0.0.0:14552', source_system=255, wait_for_heartbeat=True)
    commander.start()
    
    # State Machine
    TARGET_ALT = 50.0
    TAKE_OFF_COORD = (-29.987222, 153.228056)
    TARGET_COORDS = [
        (-29.985, 153.225),
        (-29.985, 153.235),
        (-29.995, 153.235),
        (-29.995, 153.225)
    ]
    
    state = "IDLE"
    waypoint_idx = 0
    
    print("\n[COMMANDER] Starting Automated Test Cycle...")
    
    try:
        while True:
            pose = commander.get_pose()
            
            if state == "IDLE":
                if pose['lat'] != 0:
                    print("[COMMANDER] Pre-arm checks pass. Setting GUIDED mode...")
                    commander.set_mode("GUIDED")
                    time.sleep(1.0)
                    print("[COMMANDER] Arming...")
                    commander.arm(True)
                    state = "ARMING"
            
            elif state == "ARMING":
                # In real scenario we'd check HEARTBEAT base_mode for MAV_MODE_FLAG_SAFETY_ARMED
                # For SITL we wait a few secs
                time.sleep(2.0)
                print(f"[COMMANDER] Taking off to {TARGET_ALT}m...")
                commander.takeoff(TARGET_ALT)
                state = "TAKEOFF"
                
            elif state == "TAKEOFF":
                if pose['alt_agl'] >= TARGET_ALT * 0.9:
                    print(f"[COMMANDER] Altitude reached. Setting GUIDED and starting pattern...")
                    commander.set_mode("GUIDED")
                    state = "FLY_PATTERN"
            
            elif state == "FLY_PATTERN":
                target_lat, target_lon = TARGET_COORDS[waypoint_idx]
                commander.goto(target_lat, target_lon, TARGET_ALT)
                
                # Check distance to waypoint
                dist_lat = abs(pose['lat'] - target_lat)
                dist_lon = abs(pose['lon'] - target_lon)
                if dist_lat < 0.0001 and dist_lon < 0.0001:
                    print(f"[COMMANDER] Reached Waypoint {waypoint_idx + 1}")
                    waypoint_idx += 1
                    if waypoint_idx >= len(TARGET_COORDS):
                        print("[COMMANDER] Pattern complete. Returning to launch...")
                        commander.set_mode("RTL")
                        state = "FINISHING"
            
            elif state == "FINISHING":
                if pose['alt_agl'] < 2.0:
                    print("[COMMANDER] Mission Complete. Disarming.")
                    commander.arm(False)
                    break
            
            print(f"[CMD] State: {state} | Alt: {pose['alt_agl']:.1f}m | WP: {waypoint_idx}", end='\r')
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[COMMANDER] Manual Override. Aborting...")
    finally:
        commander.running = False
        print("[COMMANDER] Shutting down.")

if __name__ == "__main__":
    main()
