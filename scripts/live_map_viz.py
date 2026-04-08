"""
Live TRN Map Visualizer
========================
Connects to ArduPilot SITL on port 14552 (separate from the main controller
on 14551 — SITL allows multiple GCS connections simultaneously).

Displays a live OpenCV window showing:
  ┌────────────────────────────────────────────────────────┐
  │  Orthophoto tile centred on GPS position               │
  │   - Yellow rectangle  = TRN search window boundary     │
  │   - Heat map overlay  = template correlation scores    │
  │     (bright = high confidence, dark = low confidence)  │
  │   - Green  dot/trail  = GPS reported position          │
  │   - Cyan   dot        = TRN matched lock position      │
  │   - Red line          = error vector GPS → TRN         │
  │   - Status bar        = PSR score, EKF status, speed   │
  └────────────────────────────────────────────────────────┘

Run this ALONGSIDE unified_controller.py — it is read-only and does NOT
send any commands to the drone.

Usage:
    $env:PYTHONPATH="."; .\venv\Scripts\python.exe scripts\live_map_viz.py
"""

import math
import time
import threading
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from pymavlink import mavutil
import os

os.environ['MAVLINK20'] = '1'

# ── Config ─────────────────────────────────────────────────────────────────
SITL_ADDR   = 'udpin:127.0.0.1:14552'   # Second GCS port — SITL allows multiple
ORTHO_PATH  = r"C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif"
ORTHO_RES   = 0.5      # m/px
HFOV_DEG    = 69.7     # DJI Mini 3 Pro
MARGIN_M    = 100.0    # Search window margin
MIN_PSR     = 0.10
WIN_SIZE    = 900      # OpenCV window size (pixels)
TRAIL_LEN   = 60       # Number of past positions to draw in trail

# ── Helpers ─────────────────────────────────────────────────────────────────
def footprint_px(agl_m, img_w=4000, img_h=2250):
    fov_w = 2.0 * agl_m * math.tan(math.radians(HFOV_DEG / 2.0))
    fov_h = fov_w * (img_h / img_w)
    return max(1, int(fov_w / ORTHO_RES)), max(1, int(fov_h / ORTHO_RES))

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

# ── Telemetry reader (background thread) ─────────────────────────────────────
class LiveTelemetry:
    def __init__(self, addr):
        print(f"[VIZ] Connecting to SITL on {addr}...")
        self.mav = mavutil.mavlink_connection(addr, source_system=254)
        self.pose = {'lat': 0, 'lon': 0, 'alt_agl': 50.0, 'yaw': 0.0,
                     'vx': 0.0, 'vy': 0.0}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            msg = self.mav.recv_match(
                type=['GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=True, timeout=1
            )
            if msg is None:
                continue
            t = msg.get_type()
            with self._lock:
                if t == 'GLOBAL_POSITION_INT':
                    self.pose['lat'] = msg.lat / 1e7
                    self.pose['lon'] = msg.lon / 1e7
                    self.pose['alt_agl'] = msg.relative_alt / 1000.0
                    self.pose['vx'] = msg.vx / 100.0
                    self.pose['vy'] = msg.vy / 100.0
                elif t == 'ATTITUDE':
                    import numpy as _np
                    self.pose['yaw'] = _np.rad2deg(msg.yaw)

    def get(self):
        with self._lock:
            return self.pose.copy()

# ── Main visualizer ───────────────────────────────────────────────────────────
def main():
    src      = rasterio.open(ORTHO_PATH)
    to_mga   = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:28356", "EPSG:4326", always_xy=True)
    telem    = LiveTelemetry(SITL_ADDR)

    # GPS trail (list of canvas pixel coords)
    gps_trail = []
    trn_trail = []

    cv2.namedWindow("TRN Live Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TRN Live Map", WIN_SIZE, WIN_SIZE + 60)

    print("[VIZ] Waiting for GPS fix...")

    while True:
        pose = telem.get()
        if pose['lat'] == 0:
            # No fix yet — show waiting screen
            blank = np.zeros((WIN_SIZE + 60, WIN_SIZE, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for SITL GPS fix...", (20, WIN_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow("TRN Live Map", blank)
            if cv2.waitKey(200) == 27:
                break
            continue

        gps_lat  = pose['lat']
        gps_lon  = pose['lon']
        agl_m    = max(pose['alt_agl'], 10.0)
        yaw_deg  = pose['yaw']
        imu_spd  = math.sqrt(pose['vx']**2 + pose['vy']**2)

        # Map coordinates
        gps_e, gps_n = to_mga.transform(gps_lon, gps_lat)
        cen_col, cen_row = ~src.transform * (gps_e, gps_n)
        cen_col, cen_row = int(cen_col), int(cen_row)

        tile_w, tile_h = footprint_px(agl_m)
        margin_px = int(MARGIN_M / ORTHO_RES)
        search_w  = tile_w + 2 * margin_px
        search_h  = tile_h + 2 * margin_px

        win = rasterio.windows.Window(
            cen_col - search_w // 2,
            cen_row - search_h // 2,
            search_w, search_h
        )
        patch = src.read(window=win)
        if patch.shape[1] < tile_h or patch.shape[2] < tile_w:
            time.sleep(0.2)
            continue

        ortho_rgba = np.transpose(patch, (1, 2, 0))
        ortho_gray = cv2.cvtColor(ortho_rgba, cv2.COLOR_RGBA2GRAY)
        ortho_rgb  = cv2.cvtColor(ortho_rgba, cv2.COLOR_RGBA2RGB)

        # Simulate virtual drone view (same approach as controller)
        virtual_frame = ortho_gray[margin_px:margin_px+tile_h,
                                   margin_px:margin_px+tile_w].copy()
        if abs(yaw_deg) > 0.1:
            M = cv2.getRotationMatrix2D((tile_w//2, tile_h//2), -yaw_deg, 1.0)
            virtual_frame = cv2.warpAffine(virtual_frame, M, (tile_w, tile_h),
                                           borderMode=cv2.BORDER_REPLICATE)

        # Canny edges
        p_edge = cv2.dilate(cv2.Canny(cv2.GaussianBlur(virtual_frame, (5,5), 0), 50, 150),
                            np.ones((3,3), np.uint8), iterations=1)
        o_edge = cv2.dilate(cv2.Canny(cv2.GaussianBlur(ortho_gray, (5,5), 0), 50, 150),
                            np.ones((3,3), np.uint8), iterations=1)

        # Template match → correlation heat map
        res = cv2.matchTemplate(o_edge, p_edge, cv2.TM_CCOEFF_NORMED)
        _, psr, _, max_loc = cv2.minMaxLoc(res)

        # ── Build display canvas ─────────────────────────────────────────
        # Scale the ortho patch to WIN_SIZE
        canvas = cv2.resize(ortho_rgb, (WIN_SIZE, WIN_SIZE))
        scale_x = WIN_SIZE / search_w
        scale_y = WIN_SIZE / search_h

        # ── Heat map overlay (shows what the algorithm is "looking at") ──
        # Normalise to 0–255 and colourise
        res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Pad heat map back to search window size
        pad_top  = 0; pad_left  = 0
        heatmap_full = np.zeros((search_h, search_w), dtype=np.uint8)
        h_h, h_w = res_norm.shape
        heatmap_full[:h_h, :h_w] = res_norm
        heatmap_color = cv2.applyColorMap(heatmap_full, cv2.COLORMAP_JET)
        heatmap_rs    = cv2.resize(heatmap_color, (WIN_SIZE, WIN_SIZE))
        # Blend 35% heat map over the ortho
        canvas = cv2.addWeighted(canvas, 0.65, heatmap_rs, 0.35, 0)

        # ── Search window boundary (yellow rectangle) ────────────────────
        sw_x1 = int(margin_px * scale_x)
        sw_y1 = int(margin_px * scale_y)
        sw_x2 = int((margin_px + tile_w) * scale_x)
        sw_y2 = int((margin_px + tile_h) * scale_y)
        cv2.rectangle(canvas, (sw_x1, sw_y1), (sw_x2, sw_y2), (0, 255, 255), 2)

        # ── GPS position (green dot, centre of canvas) ───────────────────
        gps_cx = WIN_SIZE // 2
        gps_cy = WIN_SIZE // 2
        gps_trail.append((gps_cx, gps_cy))
        if len(gps_trail) > TRAIL_LEN:
            gps_trail.pop(0)
        for i in range(1, len(gps_trail)):
            alpha = int(80 + 175 * i / len(gps_trail))
            cv2.line(canvas, gps_trail[i-1], gps_trail[i], (0, alpha, 0), 2)
        cv2.circle(canvas, (gps_cx, gps_cy), 8, (0, 255, 0), -1)
        cv2.circle(canvas, (gps_cx, gps_cy), 8, (255, 255, 255), 1)

        # ── TRN match position ───────────────────────────────────────────
        trn_cx = int((max_loc[0] + tile_w // 2) * scale_x)
        trn_cy = int((max_loc[1] + tile_h // 2) * scale_y)

        # EKF plausibility check (simplified, same logic as controller)
        trn_e = gps_e + (max_loc[0] - margin_px) * ORTHO_RES
        trn_n = gps_n - (max_loc[1] - margin_px) * ORTHO_RES
        trn_lon, trn_lat = to_wgs84.transform(trn_e, trn_n)
        err_m = haversine_m(gps_lat, gps_lon, trn_lat, trn_lon)

        if psr >= MIN_PSR:
            # Draw error vector GPS → TRN
            cv2.arrowedLine(canvas, (gps_cx, gps_cy), (trn_cx, trn_cy),
                            (0, 0, 255), 2, tipLength=0.15)
            # TRN dot
            dot_col = (0, 255, 255)   # cyan = accepted
            trn_trail.append((trn_cx, trn_cy))
            if len(trn_trail) > TRAIL_LEN:
                trn_trail.pop(0)
            for i in range(1, len(trn_trail)):
                alpha = int(80 + 175 * i / len(trn_trail))
                cv2.line(canvas, trn_trail[i-1], trn_trail[i], (0, alpha, alpha), 2)
            cv2.circle(canvas, (trn_cx, trn_cy), 7, dot_col, -1)
            cv2.circle(canvas, (trn_cx, trn_cy), 7, (255,255,255), 1)
            lock_str = f"LOCK  {err_m:.1f}m"
        else:
            lock_str = f"LOW CONF"
            trn_trail.clear()

        # ── Status bar ───────────────────────────────────────────────────
        status_bar = np.zeros((60, WIN_SIZE, 3), dtype=np.uint8)
        bar_col = (0, 220, 0) if psr >= MIN_PSR else (0, 140, 255)
        info = (f"  GPS: {gps_lat:.6f}, {gps_lon:.6f}  |  "
                f"Alt: {agl_m:.1f}m  |  Yaw: {yaw_deg:.1f}°  |  "
                f"IMU: {imu_spd:.1f}m/s  |  PSR: {psr:.3f}  |  {lock_str}")
        cv2.putText(status_bar, info, (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, bar_col, 1, cv2.LINE_AA)

        display = np.vstack([canvas, status_bar])
        cv2.imshow("TRN Live Map", display)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or Q to quit
            break

        time.sleep(0.25)   # ~4 fps — enough for live monitoring

    cv2.destroyAllWindows()
    src.close()
    print("[VIZ] Closed.")

if __name__ == "__main__":
    main()
