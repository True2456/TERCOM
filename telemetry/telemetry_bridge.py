from pymavlink import mavutil
import time
import threading
import numpy as np
import os

# Force MAVLink 2.0 for ArduPilot SITL compatibility
os.environ['MAVLINK20'] = '1'

class TelemetryBridge:
    def __init__(self, connection_string='udpout:127.0.0.1:14551', source_system=255, wait_for_heartbeat=True):
        print(f"Connecting to SITL on {connection_string}...")
        self.mav = mavutil.mavlink_connection(connection_string, source_system=source_system)
        
        # Pre-emptive handshake to "wake up" some SITL environments
        print("Sending proactive handshake...")
        self.mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, 
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
        )
        
        if wait_for_heartbeat:
            print("Waiting for SITL heartbeat...")
            # Add timeout to avoid permanent hang
            hb = self.mav.wait_heartbeat(timeout=20)
            if hb is None:
                print("[WARNING] No heartbeat received, but proceeding anyway...")
            else:
                print(f"Heartbeat received from System {self.mav.target_system}!")
        else:
            print("Heartbeat wait skipped. Listening...")
        
        self.latest_data = {
            'lat': 0,
            'lon': 0,
            'alt_agl': 0,
            'pitch': 0,
            'roll': 0,
            'yaw': 0,
            'time_boot_ms': 0,
            'armed': False,
            # Ground speed components (cm/s from MAVLink, converted to m/s)
            'vx': 0.0,  # North velocity m/s
            'vy': 0.0,  # East velocity m/s
            'vz': 0.0,  # Down velocity m/s
        }
        
        self.running = False
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()
        print("Telemetry listener started.")
    def stop(self):
        self.running = False
        print("Telemetry listener stopping...")

    def _listen_loop(self):
        last_heartbeat = 0
        while self.running:
            try:
                # Send Heartbeat at 1Hz
                if time.time() - last_heartbeat > 1.0:
                    self.mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS, 
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
                    last_heartbeat = time.time()

                msg = self.mav.recv_match(
                    type=['GLOBAL_POSITION_INT', 'ATTITUDE', 'DISTANCE_SENSOR', 'HEARTBEAT'], 
                    blocking=False
                )
                if msg is None:
                    time.sleep(0.01)
                    continue
            except ConnectionResetError:
                # Standard WinUDP quirk: ignore and retry
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"[TELEMETRY] Listener error: {e}")
                time.sleep(0.1)
                continue
                
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                self.latest_data['lat'] = msg.lat / 1e7
                self.latest_data['lon'] = msg.lon / 1e7
                self.latest_data['time_boot_ms'] = msg.time_boot_ms
                # Ground velocities (vx/vy/vz in cm/s -> m/s)
                self.latest_data['vx'] = msg.vx / 100.0
                self.latest_data['vy'] = msg.vy / 100.0
                self.latest_data['vz'] = msg.vz / 100.0
                # Fallback altitude if DISTANCE_SENSOR is missing
                if self.latest_data.get('alt_agl', 0) == 0:
                    self.latest_data['alt_agl'] = msg.relative_alt / 1000.0 # mm to m

            elif msg.get_type() == 'ATTITUDE':
                self.latest_data['pitch'] = np.rad2deg(msg.pitch)
                self.latest_data['roll'] = np.rad2deg(msg.roll)
                self.latest_data['yaw'] = np.rad2deg(msg.yaw)
            elif msg.get_type() == 'DISTANCE_SENSOR':
                self.latest_data['alt_agl'] = msg.current_distance / 100.0 # cm to m
            elif msg.get_type() == 'HEARTBEAT':
                self.latest_data['armed'] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

    def send_gps_input(self, lat, lon, alt, n_sats=10, fix_type=3, gps_id=14):
        """
        Injects a Virtual GPS message (GPS_INPUT) into ArduPilot.
        Now using ID 14 as requested.
        """
        self.mav.mav.gps_input_send(
            0,          # Timestamp (0 for auto)
            gps_id,     # ID (Changed to 14)
            8 | 16 | 32, # Flags: Use Lat/Lon, Alt, HDOP
            int(time.time() * 1e3) % 4294967295, # time_week_ms (crude 32-bit wrap)
            0,          # time_week
            fix_type,   # fix_type (3 = 3D Fix)
            int(lat * 1e7), 
            int(lon * 1e7), 
            alt, 
            1.0,        # hdop
            1.0,        # vdop
            0, 0, 0,    # Velocity VN, VE, VD
            1.0,        # Speed accuracy
            1.0,        # Horiz accuracy
            1.0,        # Vert accuracy
            n_sats      # Satellites visible
        )

    def send_vision_estimate(self, lat, lon, alt, roll, pitch, yaw, confidence=1.0):
        """
        Sends VISION_POSITION_ESTIMATE to ArduPilot.
        """
        # ArduPilot expects: usec, x, y, z, roll, pitch, yaw
        # For TRN, we send Global Position if possible, or use ODOMETRY.
        # VISION_POSITION_ESTIMATE uses (x, y, z) in meters relative to EKF origin.
        # But we can also use GPS_INPUT if we want to spoof a GPS.
        # However, VISION_POSITION_ESTIMATE is standard for "Vision-based" positioning.
        
        # Note: ArduPilot's EKF3 can fuse VISION_POSITION_ESTIMATE as absolute position 
        # if the coordinate frame is handled.
        
        self.mav.mav.vision_position_estimate_send(
            int(time.time() * 1e6), # usec
            lat, # Actually x (m) - wait, message says x,y,z
            lon, # actually y (m)
            alt, # actually z (m)
            np.deg2rad(roll),
            np.deg2rad(pitch),
            np.deg2rad(yaw),
            [0]*21, # Covariance
            0 # Reset counter
        )
        # print(f"Sent Vision Estimate: {lat}, {lon} (Confidence: {confidence})")

    def get_pose(self):
        return self.latest_data.copy()

    def set_mode(self, mode_name):
        """
        Sets the flight mode (e.g., GUIDED, AUTO, RTL).
        """
        if mode_name not in self.mav.mode_mapping():
            print(f"Unknown mode: {mode_name}")
            return
        mode_id = self.mav.mode_mapping()[mode_name]
        self.mav.set_mode(mode_id)
        print(f"Set mode to {mode_name}")

    def arm(self, status=True):
        """
        Arms (True) or Disarms (False) the vehicle.
        """
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1 if status else 0,
            0, 0, 0, 0, 0, 0
        )
        print(f"Sent {'ARM' if status else 'DISARM'} command.")

    def takeoff(self, altitude):
        """
        Sends a takeoff command to the specified altitude (m).
        """
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0,
            altitude
        )
        print(f"Sent TAKEOFF command to {altitude}m.")

    def goto(self, lat, lon, alt):
        """
        Sends a GUIDED goto command.
        """
        # Using the vehicle's latest reported time_boot_ms to prevent overflow
        self.mav.mav.set_position_target_global_int_send(
            self.latest_data['time_boot_ms'],
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000, # Position only
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0, # Velocity
            0, 0, 0, # Accel
            0, 0     # Yaw
        )

if __name__ == "__main__":
    # Test connection
    bridge = TelemetryBridge()
    bridge.start()
    try:
        while True:
            pose = bridge.get_pose()
            print(f"Pose: Lat={pose['lat']:.6f}, Lon={pose['lon']:.6f}, Alt={pose['alt_agl']:.1f}m", end='\r')
            time.sleep(0.5)
    except KeyboardInterrupt:
        bridge.running = False
        print("\nStopped.")
