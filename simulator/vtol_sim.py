import time
import math
import json
import threading
import socket
import os
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil

class TailsitterSim:
    def __init__(self, config_path=None):
        if config_path is None:
            # Locate relative to script 🚀
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "sim_config.json")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.origin = self.config["origin"]
        self.drone_cfg = self.config["drone"]
        self.net_cfg = self.config["network"]
        
        # State Vectors (Atomic Access)
        self.lat = self.origin["lat"]
        self.lon = self.origin["lon"]
        self.alt = self.origin["alt"]
        self.roll = 0.0
        self.pitch = 0.0 # Vertical hover = 90 deg? No, ArduPilot uses 0 for level hover in VTOL mode?
        # Standard convention: Pitch 0 = Horizon. Vertical Hover = Pitch 90.
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.batt_pct = 98
        self.is_armed = False
        self.mode = "QSTABILIZE"
        self.mission_active = False
        self.current_waypoint = 0
        self.waypoints = []
        self.is_transitioned = False # Track mission phase 🛰️
        self.gps_enabled = True # Simulator GPS Active Flag 🛰️
        self.gps2_enabled = True # Simulator TRN / GPS2 Active Flag 🛰️
        self.gps2_data = None # Store incoming TRN fixes 🛰️
        
        self.lock = threading.Lock()
        self.mode_map = {
            "STABILIZE": 0, "CIRCLE": 1, "GUIDED": 4, "FBWA": 5, "AUTO": 10, "RTL": 11,
            "LOITER": 12, "TAKEOFF": 13, "TRANSITION": 14, "QSTABILIZE": 17,
            "QHOVER": 18, "QLOITER": 19, "QLAND": 20, "QRTL": 21
        }
        self.target_lat = self.lat
        self.target_lon = self.lon
        self.target_alt = self.alt
        
        self.last_hb = 0
        self.target_speed_ms = self.drone_cfg["cruise_speed_ms"]
        
        # Multi-Link Configuration: Active Broadcast to multiple GCS Sinks 🛰️
        # Port 1: Primary GCS (14550)
        self.sysid = self.drone_cfg["sysid"]
        self.gcs_target = (self.net_cfg["gcs_ip"], self.net_cfg["gcs_port"])
        self.nav_port = self.net_cfg["listen_port"] # Port where Visual Nav expects to receive 🛰️
        
        self.conns = []
        # Listen for connections on standard ports
        self.conns.append(mavutil.mavlink_connection(f"udpin:0.0.0.0:14550", source_system=self.sysid))
        self.conns.append(mavutil.mavlink_connection(f"udpin:0.0.0.0:14551", source_system=self.sysid))
        self.conns.append(mavutil.mavlink_connection(f"udpin:0.0.0.0:14552", source_system=self.sysid))
        
        self.running = True
        self.start_time = time.time()
        self.last_update = self.start_time
        self.last_hb = 0
        
    def _get_boot_ms(self):
        return int((time.time() - self.start_time) * 1000)

    def _get_distance_metres(self, lat1, lon1, lat2, lon2):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        return math.sqrt((dlat*dlat) + (dlon*dlon)) * 1.113195e5

    def _get_bearing(self, lat1, lon1, lat2, lon2):
        off_x = lon2 - lon1
        off_y = lat2 - lat1
        bearing = 90.0 + math.degrees(math.atan2(-off_y, off_x))
        if bearing < 0: bearing += 360.0
        return bearing
        
    def run(self):
        print(f"Mission SITL: Twister Dual-Engine Simulator Active at {self.lat}, {self.lon}")
        print(f"MAVLink 2.0 Broadcast: {self.net_cfg['gcs_ip']}:{self.net_cfg['gcs_port']}")
        
        # Listen for GCS Commands in background
        threading.Thread(target=self._recv_loop, daemon=True).start()
        
        while self.running:
            now = time.time()
            dt = now - self.last_update
            self.last_update = now
            
            self._update_physics(dt)
            self._broadcast_telemetry()
            
            time.sleep(0.05) # 20Hz Loop
            
    def _update_physics(self, dt):
        if not self.is_armed:
            # Force absolute zero on all vectors when disarmed 🛰️
            self.vx = 0.0
            self.vy = 0.0
            self.vz = 0.0
            self.roll = 0.0
            self.pitch = 0.0
            return
            
        # Basic VTOL Physics for Mission Testing
        if self.mode in ["AUTO", "TAKEOFF", "GUIDED"]:
            target_alt = self.drone_cfg["transition_alt_m"]
            
            if self.alt >= target_alt:
                self.is_transitioned = True

            if not self.is_transitioned:
                # 1. Vertical Climb Mode
                if self.vz == 0: print(f"[PHYSICS] Mode: {self.mode} | Action: Vertical Takeoff to {target_alt}m", flush=True)
                self.vz = self.drone_cfg["climb_rate_ms"]
                self.pitch = 0.0 # Standard Level Climb
                self.alt += self.vz * dt
            else:
                # 2. Waypoint Navigation
                self.vz = 0.0
                
                # --- MISSION ENGINE: Consume non-navigation commands instantly ---
                while len(self.waypoints) > 0 and self.current_waypoint < len(self.waypoints):
                    wp = self.waypoints[self.current_waypoint]
                    cmd = wp.command if hasattr(wp, 'command') else wp.get('command', 0)
                    
                    if cmd == 178: # MAV_CMD_DO_CHANGE_SPEED
                        new_speed = wp.param2 if hasattr(wp, 'param2') else wp.get('param2', self.target_speed_ms)
                        if new_speed > 0: self.target_speed_ms = new_speed
                        print(f"Mission SITL: [EXEC] DO_CHANGE_SPEED -> {self.target_speed_ms} m/s", flush=True)
                        self.current_waypoint += 1
                        continue # Re-check next item in same frame
                    elif cmd in [16, 22, 82, 84, 85]: # NAV_WAYPOINT, NAV_TAKEOFF, NAV_VTOL_TAKEOFF...
                        break
                    else:
                        # Skip other non-positional items (Condition, Jump, etc.)
                        self.current_waypoint += 1

                if len(self.waypoints) > 0 and self.current_waypoint < len(self.waypoints):
                    wp = self.waypoints[self.current_waypoint]
                    target_lat = wp.x / 1e7 if hasattr(wp, 'x') else wp['lat']
                    target_lon = wp.y / 1e7 if hasattr(wp, 'y') else wp['lon']
                    target_alt_msl = wp.z if hasattr(wp, 'z') else wp['alt']
                    
                    dist = self._get_distance_metres(self.lat, self.lon, target_lat, target_lon)
                    
                    if dist < 5.0: # 5m Arrival Radius
                        print(f"Mission SITL: Waypoint {self.current_waypoint} reached! (Dist: {dist:.1f}m)", flush=True)
                        with self.lock:
                            for conn in self.conns:
                                conn.mav.mission_item_reached_send(self.current_waypoint)
                        self.current_waypoint += 1
                        
                        # 🛰️ Mission End Guard: Transition to LOITER at final WP
                        if self.current_waypoint >= len(self.waypoints):
                            print("Mission SITL: [AUTO-RECOVERY] Final Waypoint hit. Entering Coordinated Orbit.", flush=True)
                            self.mode = "LOITER"
                            self.target_lat = target_lat
                            self.target_lon = target_lon
                        return 
                    
                    # --- FLIGHT DYNAMICS: Coordinated Turn Model ---
                    target_bearing = self._get_bearing(self.lat, self.lon, target_lat, target_lon)
                    diff = (target_bearing - self.yaw + 180) % 360 - 180
                    
                    # Limit turn rate based on config (Realistic arcs)
                    max_step = self.drone_cfg["max_yaw_rate_deg"] * dt
                    step = max(-max_step, min(max_step, diff * 0.1)) # Smoothly move but cap rate
                    
                    prev_yaw = self.yaw
                    self.yaw = (self.yaw + step) % 360
                    
                    # Coordinated Banking (Roll): Proportional to turn rate
                    turn_rate_s = step / dt
                    # INVERTED for GCS HUD Calibration 🛰️
                    # APPLY LERP (0.1 Alpha) to dampen high-frequency vibration 🛰️
                    raw_target_roll = - (turn_rate_s * self.drone_cfg["bank_factor"])
                    raw_target_roll = max(-self.drone_cfg["max_roll_deg"], min(self.drone_cfg["max_roll_deg"], raw_target_roll))
                    self.roll = (self.roll * 0.9) + (raw_target_roll * 0.1)
                    
                    self.pitch = 0.0 # Level cruise
                    
                    # Apply Speed Clamping (15-35 m/s)
                    speed = max(self.drone_cfg["min_speed_ms"], min(self.drone_cfg["max_speed_ms"], self.target_speed_ms))
                    
                    # Directional Velocity based on current Yaw
                    self.vx = speed * math.cos(math.radians(90 - self.yaw))
                    self.vy = speed * math.sin(math.radians(90 - self.yaw))
                    
                    # Update lat/lon
                    self.lat += (self.vy * dt) / 111111.0
                    self.lon += (self.vx * dt) / (111111.0 * math.cos(math.radians(self.lat)))
                    
                    # Vertical correction (move toward target altitude)
                    if self.alt < target_alt_msl - 1: self.alt += 2.0 * dt
                    elif self.alt > target_alt_msl + 1: self.alt -= 2.0 * dt
                else:
                    # MISSION COMPLETE FALLBACK: Coordinated Loiter and Pitch Level 🛰️
                    self.mode = "LOITER"
                    self.pitch = 0.0
                    self.vx = 0.0
                    self.vy = 0.0
                    print(f"[PHYSICS] Mission Complete | Mode: {self.mode} | Target: Last Waypoint", flush=True)
        elif self.mode in ["FBWA", "CIRCLE", "LOITER"]:
            # Level high-speed horizontal cruise or orbiting 🛰️
            self.pitch = 0.0
            self.vz = 0.0
            speed = self.drone_cfg["cruise_speed_ms"]
            
            if self.mode in ["CIRCLE", "LOITER"]:
                # ArduPilot-Grade Proportional Loiter Steering (Tangent Merge) 🛰️
                # We calculate a target bearing that leads the drone onto the loiter radius.
                target_radius = 50.0 
                dist = self._get_distance_metres(self.lat, self.lon, self.target_lat, self.target_lon)
                bearing_to_center = self._get_bearing(self.lat, self.lon, self.target_lat, self.target_lon)
                
                # [CENTER GUARD] If near center, exit North to break deadzone
                if dist < 0.1: 
                    bearing_to_center = 0.0
                
                # Steer Proportionally: 0-degree tangent offset at radius, 90-degree outward offset at center
                # This creates a perfect log-spiral merge into the orbit.
                # error = dist - target_radius
                # offset = 90 + (error * Kp)
                if dist < target_radius:
                    # Inside the ring: Steering out (e.g. 135 degrees relative to center)
                    target_yaw = (bearing_to_center + 135) % 360
                elif dist > target_radius + 5:
                    # Outside the ring: Steering in (e.g. 45 degrees relative to center)
                    target_yaw = (bearing_to_center + 45) % 360
                else:
                    # Near the ring: Transition to pure tangent (90 degrees)
                    target_yaw = (bearing_to_center + 90) % 360
                
                # Coordinated Turn Physics with Smoothing (Dampened Inertia) 🛰️
                old_yaw = self.yaw
                yaw_diff = (target_yaw - old_yaw + 180) % 360 - 180
                max_step = self.drone_cfg["max_yaw_rate_deg"] * dt
                step = max(-max_step, min(max_step, yaw_diff * 0.1)) # Dampened step
                
                self.yaw = (self.yaw + step) % 360
                
                # INVERTED for GCS HUD Calibration 🛰️
                # APPLY LERP (0.2 Alpha) to match physical UAV responsiveness 🛰️
                turn_rate_s = step / dt
                raw_target_roll = - (turn_rate_s * self.drone_cfg["bank_factor"])
                raw_target_roll = max(-self.drone_cfg["max_roll_deg"], min(self.drone_cfg["max_roll_deg"], raw_target_roll))
                self.roll = (self.roll * 0.8) + (raw_target_roll * 0.2)
            else:
                self.roll = 0.0 # Level FBWA / Straight Recovery 🛰️
                self.pitch = 0.0 
                # Maintain heading or default to North if mission just completed
                if self.vx == 0 and self.vy == 0:
                    self.yaw = 0.0
            
            # Use cruise speed for standard orbit
            current_orbit_speed = self.drone_cfg["cruise_speed_ms"]
            self.vx = current_orbit_speed * math.cos(math.radians(90 - self.yaw))
            self.vy = current_orbit_speed * math.sin(math.radians(90 - self.yaw))
            self.lat += (self.vy * dt) / 111111.0
            self.lon += (self.vx * dt) / (111111.0 * math.cos(math.radians(self.lat)))
        elif self.mode in ["QLOITER", "QRTL", "QSTABILIZE", "TAKEOFF", "TRANSITION", "RTL"]:
            # Hover, Vertical Climb, or Transition 🛰️
            self.vz = 0.0
            
            if self.mode == "TAKEOFF":
                climb_rate = 2.2 # 2.2m/s realistic heavy climb 🛰️
                self.vx = 0.0
                self.vy = 0.0
                self.pitch = 0.0
                if self.alt < self.target_alt:
                    # Pure Vertical Climb
                    self.alt += climb_rate * dt
                else:
                    self.mode = "TRANSITION"
                    print(f"Mission SITL: Reach Takeoff Alt ({self.target_alt:.1f}m). Initiating Forward Transition...", flush=True)
            
            elif self.mode in ["QRTL", "RTL"]:
                # 1. Return Home Logic 🛰️
                dist = self._get_distance_metres(self.lat, self.lon, self.origin["lat"], self.origin["lon"])
                
                if dist > 45.0:
                    # Fly toward home (horizontal) 🛰️
                    self.yaw = self._get_bearing(self.lat, self.lon, self.origin["lat"], self.origin["lon"])
                    speed = self.drone_cfg["cruise_speed_ms"]
                    self.vx = speed * math.cos(math.radians(90 - self.yaw))
                    self.vy = speed * math.sin(math.radians(90 - self.yaw))
                    self.pitch = 0.0
                    self.roll = 0.0 # Forced Level Rollout during Return 🛰️
                elif self.mode == "RTL":
                    # RTL: Station-keeping Orbit (Fixed-Wing) 🛰️
                    self.mode = "LOITER"
                    self.target_lat = self.origin["lat"]
                    self.target_lon = self.origin["lon"]
                    print("Mission SITL: [RTL] Home reached. Entering Station-Keeping Orbit.", flush=True)
                else:
                    # QRTL: Back-Transition Flare (Pitch-Up) 🛰️
                    # Flare at 15 deg/sec up to 90
                    target_pitch = 90.0
                    if self.pitch < target_pitch:
                        self.pitch += 15.0 * dt
                        # Horizontal air-braking (Drag)
                        current_speed = math.sqrt(self.vx**2 + self.vy**2)
                        bleed = 5.0 * dt
                        if current_speed > 0:
                            ratio = (max(0, current_speed - bleed)) / current_speed
                            self.vx *= ratio
                            self.vy *= ratio
                    
                    if self.pitch >= 85.0 and math.sqrt(self.vx**2 + self.vy**2) < 2.0:
                        # 2. Arrived & Vertical: Initiate Landing Sequence 🛰️
                        self.vx = 0.0
                        self.vy = 0.0
                        self.pitch = 90.0
                        
                        # --- REALSITIC LANDING FLARE ---
                        descent_rate = 1.5 # Standard descent
                        if self.alt < 2.0:
                            descent_rate = 0.5 # Flare for soft touchdown
                        
                        if self.alt > 0.05:
                            self.alt -= descent_rate * dt
                        else:
                            self.alt = 0.0
                            self.is_armed = False # Auto-disarm on touchdown
                            self.mode = "QSTABILIZE"
                            print("Mission SITL: TOUCHDOWN (Soft Flare)! Recovery complete.", flush=True)

            elif self.mode == "TRANSITION":
                # Tilt forward and accelerate 🛰️
                target_pitch = -45.0 # Dramatic tailsitter tilt
                if self.pitch > target_pitch:
                    self.pitch -= 15.0 * dt
                    # --- TRANSITION SETTLE ---
                    # Lose vertical lift before airspeed generates wing lift
                    self.alt -= 2.5 * dt
                
                # Vectorized Acceleration based on current Yaw 🛰️
                speed_accel = 5.0 * dt
                self.vx += speed_accel * math.cos(math.radians(90 - self.yaw))
                self.vy += speed_accel * math.sin(math.radians(90 - self.yaw))
                current_speed = math.sqrt(self.vx**2 + self.vy**2)
                
                if current_speed >= 15.0:
                    self.mode = "LOITER"
                    self.pitch = 0.0 # Return to cruise pitch
                    print(f"Mission SITL: Transition complete ({current_speed:.1f} m/s). Entering Orbitring.", flush=True)
            
            elif self.mode in ["QLOITER", "QRTL", "QSTABILIZE"]:
                self.vx = 0.0
                self.vy = 0.0
                self.pitch = 0.0
                if self.alt < 20: self.alt += 2.0 * dt
            
            # Update lat/lon
            self.lat += (self.vy * dt) / 111111.0
            self.lon += (self.vx * dt) / (111111.0 * math.cos(math.radians(self.lat)))
        elif self.mode == "GUIDED":
            # 🚀 Forward Flight Navigation Model for GUIDED mode
            target_lat = self.target_lat
            target_lon = self.target_lon
            target_alt = self.target_alt
            
            # 1. Heading Control (Coordinated Steering)
            target_bearing = self._get_bearing(self.lat, self.lon, target_lat, target_lon)
            diff = (target_bearing - self.yaw + 180) % 360 - 180
            max_step = self.drone_cfg["max_yaw_rate_deg"] * dt
            step = max(-max_step, min(max_step, diff * 0.1))
            self.yaw = (self.yaw + step) % 360
            
            # 2. Bank & Speed (Maintain Cruise)
            turn_rate_s = step / dt
            raw_target_roll = - (turn_rate_s * self.drone_cfg["bank_factor"])
            self.roll = (self.roll * 0.9) + (raw_target_roll * 0.1)
            self.pitch = 0.0
            
            speed = self.drone_cfg["cruise_speed_ms"]
            self.vx = speed * math.cos(math.radians(90 - self.yaw))
            self.vy = speed * math.sin(math.radians(90 - self.yaw))
            
            # 3. Position Update
            self.lat += (self.vy * dt) / 111111.0
            self.lon += (self.vx * dt) / (111111.0 * math.cos(math.radians(self.lat)))
            
            # 4. Altitude maintenance (Baro logic)
            if self.alt < target_alt - 1: self.alt += 2.0 * dt
            elif self.alt > target_alt + 1: self.alt -= 2.0 * dt
        else:
            # Other modes: No movement for now
            pass
            
    def _broadcast_telemetry(self):
        # 1. Heartbeat (1Hz)
        now = time.time()
        with self.lock:
            if now - self.last_hb >= 1.0:
                self.last_hb = now
                custom_mode = self.mode_map.get(self.mode, 0)
                base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                if self.is_armed:
                    base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                
                for conn in self.conns:
                    conn.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_FIXED_WING, 
                        mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                        base_mode,
                        custom_mode,
                        mavutil.mavlink.MAV_STATE_ACTIVE if self.is_armed else mavutil.mavlink.MAV_STATE_STANDBY
                    )
                    # Mission status broadcast
                    if len(self.waypoints) > 0:
                        conn.mav.mission_current_send(self.current_waypoint)
                
                print(f"Mission SITL: Heartbeat dispatched (Mode: {self.mode}) [ACTIVE-MULTI]", flush=True)
                
            # 2. System Status (2Hz)
            for conn in self.conns:
                conn.mav.sys_status_send(
                    0, 0, 0, 500, int(12400), # 12.4V
                    self.batt_pct, 0, 0, 0, 0, 0, 0, 0
                )
        
        with self.lock:
            # 3. Attitude (10Hz)
            for conn in self.conns:
                conn.mav.attitude_send(
                    self._get_boot_ms(),
                    math.radians(self.roll),
                    math.radians(self.pitch),
                    math.radians(self.yaw),
                    0, 0, 0
                )
            
            if self.gps_enabled:
                # 4.1 GPS_RAW_INT (10Hz) - Primary GPS
                for conn in self.conns:
                    conn.mav.gps_raw_int_send(
                        self._get_boot_ms() * 1000, 3, 
                        int(self.lat * 1e7), int(self.lon * 1e7), int(self.alt * 1000), 
                        100, 100, 0, 0, 10
                    )
                
            # 4.1.2 GLOBAL_POSITION_INT (10Hz) - Fused Position for Map
            # This is critical for the GCS map update 🛰️
            if self.gps_enabled or (self.gps2_enabled and self.gps2_data):
                # Use TRN position if primary GPS is disabled
                out_lat = self.gps2_data['lat'] if (not self.gps_enabled and self.gps2_data) else self.lat
                out_lon = self.gps2_data['lon'] if (not self.gps_enabled and self.gps2_data) else self.lon
                
                for conn in self.conns:
                    conn.mav.global_position_int_send(
                        self._get_boot_ms(),
                        int(out_lat * 1e7), int(out_lon * 1e7),
                        int(self.alt * 1000),      # MSL Alt
                        int(self.alt * 1000),      # Rel Alt
                        int(self.vx * 100), int(self.vy * 100), int(self.vz * 100),
                        int(self.yaw * 100)
                    )
            
            # 4.2 GPS2 / EXTERN RELAY (GPS_INPUT #232 or GPS_RAW_INT #24) 🛰️
            if self.gps2_enabled and self.gps2_data:
                for conn in self.conns:
                    # Relay the incoming TRN position to the GCS as GPS2
                    conn.mav.gps2_raw_int_send(
                        self._get_boot_ms() * 1000, 
                        int(self.gps2_data['fix_type']),
                        int(self.gps2_data['lat'] * 1e7), 
                        int(self.gps2_data['lon'] * 1e7), 
                        int(self.gps2_data['alt'] * 1000), 
                        100, 100, 0, 0, 
                        int(self.gps2_data['satellites_visible'])
                    )
            
            # 4.3 EKF_STATUS_REPORT (2Hz) 🛰️
            # Health depends on GPS2 (TRN) in this simulated tactical mode 🛰️
            ekf_flags = 8 if self.gps2_enabled else 0
            for conn in self.conns:
                conn.mav.ekf_status_report_send(
                    ekf_flags, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
                )
            
            # 5. VFR_HUD (5Hz)
            # Decoupled from GPS status to simulate Baro & Airspeed sensor suite 🛰️
            spd_2d = math.sqrt(self.vx**2 + self.vy**2)
            for conn in self.conns:
                conn.mav.vfr_hud_send(
                    spd_2d,      # Airspeed (Independent Sensor)
                    spd_2d,      # Groundspeed
                    int(self.yaw),
                    50,          # Throttle (Assume cruising)
                    self.alt,    # Altitude (Barometric / SF20 Lidar)
                    -self.vz     # Climb rate (inverted for VFR_HUD convention)
                )
                
                # 6. DISTANCE_SENSOR (10Hz): Simulate LightWare SF20 Lidar Rangefinder 🛰️
                # Sends relative altitude to ground at high frequency for terrain following.
                # current_distance is uint16 (0 to 65535 cm)
                dist_cm = max(0, min(65535, int(self.alt * 100)))
                conn.mav.distance_sensor_send(
                    self._get_boot_ms(),
                    10,          # Min Distance (cm)
                    10000,       # Max Distance (cm)
                    dist_cm,
                    mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,
                    1,           # Sensor ID
                    mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270, # Downward
                    255          # Covariance (Unknown)
                )

                # 7. NAV_CONTROLLER_OUTPUT (5Hz): Track mission waypoint distance 🛰️
                wp_dist = 0.0
                if self.mode == "AUTO" and self.current_waypoint < len(self.waypoints):
                    wp = self.waypoints[self.current_waypoint]
                    t_lat = wp.x / 1e7 if hasattr(wp, 'x') else wp['lat']
                    t_lon = wp.y / 1e7 if hasattr(wp, 'y') else wp['lon']
                    wp_dist = self._get_distance_metres(self.lat, self.lon, t_lat, t_lon)
                
                # Clamp wp_dist to uint16 (0 to 65535 meters) 🛡️
                safe_wp_dist = max(0, min(65535, int(wp_dist)))
                conn.mav.nav_controller_output_send(
                    0, 0, int(self.yaw), int(self.yaw), safe_wp_dist, 0, 0, 0
                )

    def _recv_loop(self):
        while self.running:
            for current_conn in self.conns:
                with self.lock:
                    try:
                        msg = current_conn.recv_match(blocking=False)
                    except Exception as e:
                        continue

                if not msg:
                    continue

                mtype = msg.get_type()
                with self.lock:
                    if mtype == 'COMMAND_LONG':
                        if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                            self.is_armed = (msg.param1 == 1)
                            if not self.is_armed:
                                self.is_transitioned = False # Reset on disarm
                            print(f"Mission SITL: Drone {'ARMED' if self.is_armed else 'DISARMED'} via MAVLink command.", flush=True)
                            current_conn.mav.command_ack_send(
                                msg.command, 
                                mavutil.mavlink.MAV_RESULT_ACCEPTED, 
                                target_system=msg.get_srcSystem(), 
                                target_component=msg.get_srcComponent()
                            )
                        elif msg.command == mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED:
                            self.target_speed_ms = msg.param2
                            print(f"Mission SITL: Target Speed updated to {self.target_speed_ms} m/s", flush=True)
                            current_conn.mav.command_ack_send(
                                msg.command, 
                                mavutil.mavlink.MAV_RESULT_ACCEPTED, 
                                target_system=msg.get_srcSystem(), 
                                target_component=msg.get_srcComponent()
                            )
                        
                        elif msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                            if self.is_armed:
                                self.mode = "TAKEOFF"
                                self.target_alt = self.origin["alt"] + msg.param7
                                self.target_lat = self.lat
                                self.target_lon = self.lon
                                print(f"Mission SITL: Initiating Vertical Takeoff to {self.target_alt}m...", flush=True)
                                current_conn.mav.command_ack_send(
                                    msg.command, 
                                    mavutil.mavlink.MAV_RESULT_ACCEPTED, 
                                    target_system=msg.get_srcSystem(), 
                                    target_component=msg.get_srcComponent()
                                )

                        elif msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                            new_mode_id = int(msg.param2 if msg.param2 != 0 else msg.param1)
                            mode_found = False
                            for m_name, m_id in self.mode_map.items():
                                if m_id == new_mode_id:
                                    self.mode = m_name
                                    mode_found = True
                                    if m_name != "AUTO":
                                        self.is_transitioned = False # Reset if leaving AUTO
                                    if m_name in ["LOITER", "CIRCLE"]:
                                        self.target_lat = self.lat
                                        self.target_lon = self.lon
                                    break
                            current_conn.mav.command_ack_send(
                                msg.command, 
                                mavutil.mavlink.MAV_RESULT_ACCEPTED, 
                                target_system=msg.get_srcSystem(), 
                                target_component=msg.get_srcComponent()
                            )
                        
                        elif msg.command == 31010: # MAV_CMD_USER_1: Custom Hack for GPS Disabling 🛰️
                            if msg.param1 != -1:
                                self.gps_enabled = (msg.param1 == 1.0)
                            if msg.param2 != -1:
                                self.gps2_enabled = (msg.param2 == 1.0)
                            
                            status1 = "ENABLED" if self.gps_enabled else "DISABLED"
                            status2 = "ENABLED" if self.gps2_enabled else "DISABLED"
                            print(f"Mission SITL: GPS1 {status1} / GPS2 {status2} via custom MAVLink command.", flush=True)
                            
                            # Broadcast status text back to HUD 🚀
                            msg_txt = f"GPS1:{status1} GPS2:{status2}"
                            current_conn.mav.statustext_send(
                                mavutil.mavlink.MAV_SEVERITY_CRITICAL if not (self.gps_enabled and self.gps2_enabled) else mavutil.mavlink.MAV_SEVERITY_INFO,
                                msg_txt.encode()
                            )
                            
                            current_conn.mav.command_ack_send(
                                msg.command, 
                                mavutil.mavlink.MAV_RESULT_ACCEPTED, 
                                target_system=msg.get_srcSystem(), 
                                target_component=msg.get_srcComponent()
                            )

                    elif mtype == 'GPS_INPUT':
                        # Store the "Virtual GPS" data from the TRN script for relay 🛰️
                        self.gps2_data = {
                            'lat': msg.lat / 1e7,
                            'lon': msg.lon / 1e7,
                            'alt': msg.alt,
                            'fix_type': msg.fix_type,
                            'satellites_visible': msg.satellites_visible
                        }
                    
                    elif mtype == 'SET_POSITION_TARGET_GLOBAL_INT':
                        # Store the target waypoint from the commander script 🛰️
                        self.target_lat = msg.lat_int / 1e7
                        self.target_lon = msg.lon_int / 1e7
                        self.target_alt = msg.alt
                        # Switch to GUIDED immediately if we were in manual/takeoff
                        if self.mode != "GUIDED":
                            self.mode = "GUIDED"
                        # print(f"Simulator: Navigation target updated to {self.target_lat}, {self.target_lon}")

                    elif mtype == 'MISSION_COUNT':
                        self.expected_count = msg.count
                        self.wp_buffer = []
                        print(f"Mission SITL: Receiving mission count -> {msg.count} items. Requesting Seq 0...", flush=True)
                        current_conn.mav.mission_request_int_send(msg.get_srcSystem(), msg.get_srcComponent(), 0)

                    elif mtype == 'MISSION_ITEM_INT':
                        self.wp_buffer.append(msg)
                        if len(self.wp_buffer) < self.expected_count:
                            current_conn.mav.mission_request_int_send(msg.get_srcSystem(), msg.get_srcComponent(), len(self.wp_buffer))
                        else:
                            self.waypoints = self.wp_buffer
                            self.current_waypoint = 0
                            current_conn.mav.mission_ack_send(msg.get_srcSystem(), msg.get_srcComponent(), mavutil.mavlink.MAV_MISSION_ACCEPTED)
                            print(f"Mission SITL: Upload complete. {len(self.waypoints)} Waypoints ready.", flush=True)

                    elif mtype == 'MISSION_SET_CURRENT':
                        self.current_waypoint = msg.seq
                        print(f"Mission SITL: Current target item set to {msg.seq}.")

                    elif mtype == 'MISSION_REQUEST_LIST':
                        current_conn.mav.mission_count_send(msg.get_srcSystem(), msg.get_srcComponent(), len(self.waypoints))

                    elif mtype == 'MISSION_REQUEST_INT':
                        if msg.seq < len(self.waypoints):
                            wp = self.waypoints[msg.seq]
                            current_conn.mav.mission_item_int_send(
                                msg.get_srcSystem(), msg.get_srcComponent(), 
                                wp.seq, wp.frame, wp.command, wp.current, wp.autocontinue,
                                wp.param1, wp.param2, wp.param3, wp.param4,
                                wp.x, wp.y, wp.z
                            )

                    elif mtype == 'SET_MODE':
                        for m_name, m_id in self.mode_map.items():
                            if m_id == msg.custom_mode:
                                self.mode = m_name
                                if m_name != "AUTO":
                                    self.is_transitioned = False # Reset if leaving AUTO
                                break
                        print(f"Mission SITL: Mode change requested -> {self.mode} (ID: {msg.custom_mode})")

            time.sleep(0.01) # Poll yield

    def _get_distance_metres(self, lat1, lon1, lat2, lon2):
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return 6371000 * c

    def _get_bearing(self, lat1, lon1, lat2, lon2):
        off_x = lon2 - lon1
        off_y = lat2 - lat1
        bearing = 90.0 + math.degrees(math.atan2(-off_y, off_x))
        return bearing % 360

    def _get_boot_ms(self):
        return int((time.time() - self.start_time) * 1000)

if __name__ == "__main__":
    sim = TailsitterSim()
    sim.run()
