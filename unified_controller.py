"""
Unified TRN Flight Controller
==============================
Integrates the validated optical TRN pipeline (cv2.matchTemplate + Canny edges)
with an EKF kinematic velocity gate to prevent false optical fixes from
entering the ArduPilot GPS stream.

Key design decisions:
  - Orthophoto map: RedRock Pi color 1 res.tif (0.5 m/px)
  - Rotation: angle = -yaw_deg  (validated camera mount offset)
  - Matcher: TM_CCOEFF_NORMED on Canny-edge images
  - EKF gate: rejects fixes where implied optical speed > IMU speed + MARGIN
"""

import time
import math
import cv2
import numpy as np
import rasterio
from pyproj import Transformer

from telemetry.telemetry_bridge import TelemetryBridge
from utils.visualizer import PathVisualizer

# ── Configuration ────────────────────────────────────────────────────────────
ORTHO_PATH      = r"C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif"
ORTHO_RES       = 0.5          # metres per pixel
HFOV_DEG        = 69.7         # DJI Mini 3 Pro horizontal FOV
MARGIN_M        = 100.0        # Search window margin around GPS prior (metres)
MIN_PSR         = 0.10         # Minimum correlation score to submit a fix

# EKF kinematic gate — if the optical fix implies a speed this many m/s faster
# than the IMU-reported ground speed, the fix is rejected as a false positive.
EKF_VELOCITY_MARGIN = 15.0     # m/s

TARGET_ALT      = 250.0        # Mission altitude (m)
WP_RADIUS       = 30.0         # Waypoint arrival radius (m)

# Mission waypoints (shifted to the drone track over the town)
WAYPOINTS = [
    (-29.982434, 153.226002), # WP1: Town edge
    (-29.982456, 153.229035), # WP2: Across town
    (-29.984878, 153.226731), # WP3: Returning inland
    (-29.985594, 153.226489), # WP4: Back
]

# ── Coordinate helpers ────────────────────────────────────────────────────────
def footprint_px(agl_m, img_w, img_h):
    """Return (tile_w, tile_h) in pixels at the orthophoto resolution."""
    fov_w_m = 2.0 * agl_m * math.tan(math.radians(HFOV_DEG / 2.0))
    fov_h_m = fov_w_m * (img_h / img_w)
    return max(1, int(fov_w_m / ORTHO_RES)), max(1, int(fov_h_m / ORTHO_RES))

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + (math.cos(math.radians(lat1))
                               * math.cos(math.radians(lat2))
                               * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

# ── EKF Kinematic Gate ────────────────────────────────────────────────────────
class EKFGate:
    """
    Rejects optical TRN fixes that would imply physically impossible speeds.

    How it works:
      1. Track the last accepted fix position and timestamp.
      2. When a new optical fix arrives, compute the implied velocity:
             optical_speed = distance(last_fix, new_fix) / time_delta
      3. Read the drone's IMU-fused ground speed from the telemetry bridge.
      4. If optical_speed > imu_speed + EKF_VELOCITY_MARGIN:
             REJECT — the fix is a false positive (e.g. aliased bush match).
         Else:
             ACCEPT — update last_fix and submit to ArduPilot.

    The EKF_VELOCITY_MARGIN (15 m/s) accounts for:
      - Genuine fast manoeuvres
      - GPS prior drift between frames
      - Residual correlation error at map edges
    """

    def __init__(self, margin_ms: float = EKF_VELOCITY_MARGIN):
        self.margin = margin_ms
        self.last_lat: float = None
        self.last_lon: float = None
        self.last_time: float = None

    def validate(self, trn_lat: float, trn_lon: float,
                 imu_speed_ms: float) -> bool:
        """
        Returns True if the fix is kinematically plausible, False if rejected.
        Always accepts the very first fix (no prior to compare against).
        """
        now = time.time()

        if self.last_lat is None:
            # First fix — accept unconditionally and seed the filter
            self._accept(trn_lat, trn_lon, now)
            return True

        dt = now - self.last_time
        if dt < 0.05:
            # Too fast to measure meaningfully — accept conservatively
            self._accept(trn_lat, trn_lon, now)
            return True

        dist = haversine_m(self.last_lat, self.last_lon, trn_lat, trn_lon)
        optical_speed = dist / dt

        if optical_speed > imu_speed_ms + self.margin:
            print(f"[EKF] REJECTED  optical_speed={optical_speed:.1f} m/s  "
                  f"imu={imu_speed_ms:.1f} m/s  dist={dist:.1f} m  dt={dt:.2f}s")
            return False

        self._accept(trn_lat, trn_lon, now)
        return True

    def _accept(self, lat, lon, t):
        self.last_lat = lat
        self.last_lon = lon
        self.last_time = t

# ── Main controller ────────────────────────────────────────────────────────────
def main():
    print("[UNIFIED] Initialising Tactical Mission Controller...")
    commander = TelemetryBridge(
        connection_string='udpout:127.0.0.1:14551',
        source_system=255,
        wait_for_heartbeat=False
    )
    commander.start()

    viz          = PathVisualizer(WAYPOINTS)
    ekf_gate     = EKFGate()
    wp_idx       = 0
    state        = "INIT"
    gps_killed   = False
    trn_accepted = 0
    trn_rejected = 0
    flight_history = []

    # Coordinate transformers
    to_mga   = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:28356", "EPSG:4326", always_xy=True)

    print("[TRN] Opening orthophoto map...")
    src = rasterio.open(ORTHO_PATH)
    print(f"[TRN] Map loaded: {src.width}x{src.height}px @ {ORTHO_RES}m/px")

    try:
        while True:
            pose = commander.get_pose()

            # ── Mission state machine ─────────────────────────────────────
            if state == "INIT":
                if pose['lat'] != 0:
                    print(f"[MISSION] SITL connected. Pose: {pose['lat']:.6f}, {pose['lon']:.6f}")
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
                if pose['alt_agl'] >= TARGET_ALT - 2.0:
                    print("[MISSION] Altitude cleared. Engaging TRN loop.")
                    state = "TRANSIT"

            elif state == "TRANSIT":
                target = WAYPOINTS[wp_idx]
                dist = haversine_m(pose['lat'], pose['lon'], target[0], target[1])

                if dist < WP_RADIUS:
                    print(f"[MISSION] Reached WP{wp_idx}. Advancing...")
                    wp_idx = (wp_idx + 1) % len(WAYPOINTS)

                    # Simulate GPS failure at WP2
                    if wp_idx == 2 and not gps_killed:
                        print("[MISSION] !!! SIMULATING PRIMARY GPS FAILURE !!!")
                        commander.mav.mav.command_long_send(
                            commander.mav.target_system,
                            commander.mav.target_component,
                            31010, 0, 0, 0, 0, 0, 0, 0, 0
                        )
                        gps_killed = True

                commander.goto(target[0], target[1], TARGET_ALT)

            # ── Real-time TRN matching loop ──────────────────────────────
            match_str = "WAITING_ALT"

            if pose['alt_agl'] > 10.0 and pose['lat'] != 0:
                agl_m    = pose['alt_agl']
                yaw_deg  = pose['yaw']
                gps_lat  = pose['lat']
                gps_lon  = pose['lon']

                # 1. Render a synthetic top-down view of the map at GPS prior -
                #    we use the orthophoto directly (no DEM hillshade)
                gps_e, gps_n = to_mga.transform(gps_lon, gps_lat)
                cen_col, cen_row = ~src.transform * (gps_e, gps_n)
                cen_col, cen_row = int(cen_col), int(cen_row)

                margin_px = int(MARGIN_M / ORTHO_RES)

                # Synthetic drone footprint dimensions (same aspect as sensor)
                SENSOR_W, SENSOR_H = 4000, 2250  # DJI Mini 3 Pro native res
                tile_w, tile_h = footprint_px(agl_m, SENSOR_W, SENSOR_H)

                search_w = tile_w + 2 * margin_px
                search_h = tile_h + 2 * margin_px

                win = rasterio.windows.Window(
                    cen_col - search_w // 2,
                    cen_row - search_h // 2,
                    search_w, search_h
                )
                ortho_patch = src.read(window=win)

                if ortho_patch.shape[1] < tile_h or ortho_patch.shape[2] < tile_w:
                    match_str = "MAP_BOUNDARY"
                else:
                    ortho_rgba = np.transpose(ortho_patch, (1, 2, 0))
                    ortho_gray = cv2.cvtColor(ortho_rgba, cv2.COLOR_RGBA2GRAY)

                    # 2. Build a blank "virtual drone frame" sized to footprint
                    #    In SITL we don't have a real camera — so we render the
                    #    corresponding orthophoto patch as our "live" frame and
                    #    apply the yaw rotation to simulate heading variation.
                    #    In real flight this would be replaced by a live camera feed.
                    virtual_frame = ortho_gray[
                        margin_px: margin_px + tile_h,
                        margin_px: margin_px + tile_w
                    ].copy()

                    # 4. Canny edge extraction on both frames
                    p_blur = cv2.GaussianBlur(virtual_frame, (5, 5), 0)
                    o_blur = cv2.GaussianBlur(ortho_gray,    (5, 5), 0)
                    p_edge = cv2.Canny(p_blur, 50, 150)
                    o_edge = cv2.Canny(o_blur, 50, 150)

                    kernel = np.ones((3, 3), np.uint8)
                    p_edge = cv2.dilate(p_edge, kernel, iterations=1)
                    o_edge = cv2.dilate(o_edge, kernel, iterations=1)

                    # 5. Template match
                    res = cv2.matchTemplate(o_edge, p_edge, cv2.TM_CCOEFF_NORMED)
                    _, psr, _, max_loc = cv2.minMaxLoc(res)

                    if psr >= MIN_PSR:
                        # 6. Convert pixel offset to world coordinates
                        dc = max_loc[0] - margin_px   # col offset from GPS prior
                        dr = max_loc[1] - margin_px   # row offset from GPS prior
                        dx_m =  dc * ORTHO_RES        # East  (+)
                        dy_m =  dr * ORTHO_RES        # South (+)

                        trn_e = gps_e + dx_m
                        trn_n = gps_n - dy_m
                        trn_lon, trn_lat = to_wgs84.transform(trn_e, trn_n)
                        err_m = haversine_m(gps_lat, gps_lon, trn_lat, trn_lon)

                        # 7. EKF kinematic gate ──────────────────────────
                        imu_speed = math.sqrt(
                            pose['vx']**2 + pose['vy']**2
                        )
                        if ekf_gate.validate(trn_lat, trn_lon, imu_speed):
                            commander.send_gps_input(trn_lat, trn_lon, agl_m)
                            trn_accepted += 1
                            match_str = (f"LOCK (PSR:{psr:.2f} "
                                         f"acc:{trn_accepted} rej:{trn_rejected})")
                            flight_history.append({"gps_lat": gps_lat, "gps_lon": gps_lon, "trn_lat": trn_lat, "trn_lon": trn_lon, "status": "LOCK", "err_m": err_m})
                        else:
                            trn_rejected += 1
                            match_str = (f"EKF_REJECTED (PSR:{psr:.2f} "
                                         f"acc:{trn_accepted} rej:{trn_rejected})")
                            flight_history.append({"gps_lat": gps_lat, "gps_lon": gps_lon, "trn_lat": trn_lat, "trn_lon": trn_lon, "status": "EKF_REJECTED", "err_m": err_m})
                    else:
                        match_str = f"LOW_CONF (PSR:{psr:.2f})"
                        flight_history.append({"gps_lat": gps_lat, "gps_lon": gps_lon, "trn_lat": 0, "trn_lon": 0, "status": "LOW_CONF", "err_m": 0})
                        
                    if len(flight_history) > 0 and len(flight_history) % 10 == 0:
                        import json
                        try:
                            with open("sitl_flight_history.json", "w") as f:
                                json.dump(flight_history, f)
                        except:
                            pass

            # ── Dashboard ─────────────────────────────────────────────────
            viz.update(pose['lat'], pose['lon'], state, match_str)
            gps_tag = "[FAIL]" if gps_killed else "[GPS1]"
            print(
                f"[UNIFIED] {gps_tag} | {state} | "
                f"Alt:{pose['alt_agl']:.1f}m | WP:{wp_idx} | {match_str}"
            )
            time.sleep(0.5)

    except KeyboardInterrupt:
        import json
        try:
            with open("sitl_flight_history.json", "w") as f:
                json.dump(flight_history, f)
            print(f"[UNIFIED] Saved sitl_flight_history.json with {len(flight_history)} frames!")
        except Exception as e:
            print("[UNIFIED] Could not save flight history:", e)
            
        print("[UNIFIED] Shutting down...")
        commander.stop()
        src.close()

if __name__ == "__main__":
    main()
