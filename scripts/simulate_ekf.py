"""
Offline EKF Gate Replay Simulator
===================================
Replays the `Test photos 3` JPEG sequence through the full unified controller
logic path (template matcher + EKF kinematic gate) without a live SITL
connection or video footage.

Each photo acts as the drone's "live camera frame" at the GPS ground truth
position. The EKF gate runs exactly as it would in flight — allowing us to
validate that it correctly accepts clean urban locks and rejects the
river/garden false positives.

Output:
  - Console: per-frame status (LOCK / EKF_REJECTED / LOW_CONF)
  - Map:     ekf_simulation_map.png with colour-coded point states
"""

import os, json, re, math, time
import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

# ── Config ─────────────────────────────────────────────────────────────────
PHOTO_DIR      = "Test photos 3"
ORTHO_PATH     = r"C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif"
GPS_TRUTH_FILE = "gps_ground_truth.json"
OUT_MAP        = "ekf_simulation_map.png"

ORTHO_RES       = 0.5   # m/px
HFOV_DEG        = 69.7  # DJI Mini 3 Pro
MARGIN_M        = 100.0
MIN_PSR         = 0.10
EKF_MARGIN_MS   = 10.0  # m/s extra allowed above IMU speed

# Simulated inter-frame interval: DJI logs show ~3 seconds between sequential photos.
# This paces the EKF velocity calculation to match real flight timing.
FRAME_INTERVAL_S = 3.0

# Simulated IMU ground speed (the drone was hovering slowly — ~2 m/s average)
# In real flight this comes from GLOBAL_POSITION_INT vx/vy.
SIM_IMU_SPEED   = 2.0   # m/s

# ── Helpers ─────────────────────────────────────────────────────────────────
def footprint_px(agl_m, img_w, img_h):
    fov_w_m = 2.0 * agl_m * math.tan(math.radians(HFOV_DEG / 2.0))
    fov_h_m = fov_w_m * (img_h / img_w)
    return max(1, int(fov_w_m / ORTHO_RES)), max(1, int(fov_h_m / ORTHO_RES))

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

# ── EKF Gate ─────────────────────────────────────────────────────────────────
class EKFGate:
    def __init__(self, margin=EKF_MARGIN_MS):
        self.margin = margin
        self.last_lat = None
        self.last_lon = None
        self.last_time = None

    def validate(self, trn_lat, trn_lon, imu_speed):
        now = time.time()
        if self.last_lat is None:
            self._accept(trn_lat, trn_lon, now)
            return True, 0.0
        dt = now - self.last_time
        if dt < 0.01:
            self._accept(trn_lat, trn_lon, now)
            return True, 0.0
        dist = haversine_m(self.last_lat, self.last_lon, trn_lat, trn_lon)
        opt_speed = dist / dt
        if opt_speed > imu_speed + self.margin:
            return False, opt_speed
        self._accept(trn_lat, trn_lon, now)
        return True, opt_speed

    def _accept(self, lat, lon, t):
        self.last_lat, self.last_lon, self.last_time = lat, lon, t

# ── Main replay ──────────────────────────────────────────────────────────────
def main():
    with open(GPS_TRUTH_FILE) as f:
        truth_db = json.load(f)

    src      = rasterio.open(ORTHO_PATH)
    to_mga   = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:28356", "EPSG:4326", always_xy=True)
    ekf      = EKFGate()

    photos = sorted(
        [p for p in os.listdir(PHOTO_DIR)
         if p.upper().endswith(".JPG") and p in truth_db]
    )

    rows = []

    print(f"\n{'Photo':<14} {'GPS Err':>8} {'PSR':>6} {'Opt Speed':>10} {'Status'}")
    print("─" * 60)

    for photo in photos:
        gps = truth_db[photo]
        gps_lat, gps_lon = gps["lat"], gps["lon"]

        img_gray = cv2.imread(os.path.join(PHOTO_DIR, photo), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        # Extract XMP metadata
        yaw_deg, agl_m = 0.0, 150.0
        with open(os.path.join(PHOTO_DIR, photo), 'rb') as fp:
            raw = fp.read(60000).decode('utf-8', 'ignore')
            y = re.findall(r'FlightYawDegree=["\']?([+-]?[0-9.]+)', raw, re.I)
            a = re.findall(r'RelativeAltitude=["\']?([+-]?[0-9.]+)', raw, re.I)
            if y: yaw_deg = float(y[0])
            if a: agl_m   = abs(float(a[0]))

        # Rotate to North-Up
        img_h, img_w = img_gray.shape
        if abs(yaw_deg) > 0.1:
            M = cv2.getRotationMatrix2D((img_w//2, img_h//2), -yaw_deg, 1.0)
            img_gray = cv2.warpAffine(img_gray, M, (img_w, img_h),
                                      borderMode=cv2.BORDER_REPLICATE)

        tile_w, tile_h = footprint_px(agl_m, img_w, img_h)

        # Build orthophoto search window
        margin_px = int(MARGIN_M / ORTHO_RES)
        search_w  = tile_w + 2 * margin_px
        search_h  = tile_h + 2 * margin_px

        gps_e, gps_n = to_mga.transform(gps_lon, gps_lat)
        cen_col, cen_row = ~src.transform * (gps_e, gps_n)
        cen_col, cen_row = int(cen_col), int(cen_row)

        win = rasterio.windows.Window(
            cen_col - search_w//2, cen_row - search_h//2,
            search_w, search_h
        )
        patch = src.read(window=win)
        if patch.shape[1] == 0 or patch.shape[2] == 0:
            continue

        ortho_gray = cv2.cvtColor(np.transpose(patch, (1,2,0)), cv2.COLOR_RGBA2GRAY)
        photo_res  = cv2.resize(img_gray, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        # Canny + dilate
        p_edge = cv2.dilate(cv2.Canny(cv2.GaussianBlur(photo_res, (5,5), 0), 50, 150),
                            np.ones((3,3), np.uint8), iterations=1)
        o_edge = cv2.dilate(cv2.Canny(cv2.GaussianBlur(ortho_gray, (5,5), 0), 50, 150),
                            np.ones((3,3), np.uint8), iterations=1)

        res = cv2.matchTemplate(o_edge, p_edge, cv2.TM_CCOEFF_NORMED)
        _, psr, _, max_loc = cv2.minMaxLoc(res)

        status = "LOW_CONF"
        trn_lat, trn_lon = gps_lat, gps_lon  # default: no update
        opt_speed = 0.0
        err_m = 0.0

        if psr >= MIN_PSR:
            dc = max_loc[0] - margin_px
            dr = max_loc[1] - margin_px
            trn_e = gps_e + dc * ORTHO_RES
            trn_n = gps_n - dr * ORTHO_RES
            trn_lon, trn_lat = to_wgs84.transform(trn_e, trn_n)
            err_m = haversine_m(gps_lat, gps_lon, trn_lat, trn_lon)
            err_m = min(err_m, 100.0)

            accepted, opt_speed = ekf.validate(trn_lat, trn_lon, SIM_IMU_SPEED)
            status = "LOCK" if accepted else "EKF_REJECTED"
        else:
            ekf.validate(gps_lat, gps_lon, SIM_IMU_SPEED)  # seed with GPS prior

        print(f"{photo:<14} {err_m:>7.1f}m  {psr:>5.2f}  "
              f"{opt_speed:>8.1f}m/s  {status}")

        rows.append(dict(
            photo=photo, gps_lat=gps_lat, gps_lon=gps_lon,
            trn_lat=trn_lat, trn_lon=trn_lon,
            err_m=err_m, psr=psr, status=status
        ))

        time.sleep(FRAME_INTERVAL_S)  # Simulate real inter-frame gap

    # ── Visual map ───────────────────────────────────────────────────────────
    if not rows:
        print("No rows to map.")
        return

    # Pick map centre and padding
    all_lats = [r['gps_lat'] for r in rows]
    all_lons = [r['gps_lon'] for r in rows]
    cen_lat  = (min(all_lats) + max(all_lats)) / 2
    cen_lon  = (min(all_lons) + max(all_lons)) / 2
    pad_m    = 200
    pad_px   = int(pad_m / ORTHO_RES)

    cen_e, cen_n = to_mga.transform(cen_lon, cen_lat)
    cen_col, cen_row = ~src.transform * (cen_e, cen_n)
    cen_col, cen_row = int(cen_col), int(cen_row)

    bg_win = rasterio.windows.Window(
        cen_col - pad_px, cen_row - pad_px, pad_px*2, pad_px*2
    )
    bg = src.read(window=bg_win)
    bg_rgb = cv2.cvtColor(np.transpose(bg, (1,2,0)), cv2.COLOR_RGBA2RGB)

    SIZE = 1200
    im   = Image.fromarray(bg_rgb).resize((SIZE, SIZE), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    sx   = SIZE / (pad_px * 2)
    sy   = SIZE / (pad_px * 2)

    def to_canvas(lat, lon):
        e, n = to_mga.transform(lon, lat)
        col, row = ~src.transform * (e, n)
        return (col - (cen_col - pad_px)) * sx, (row - (cen_row - pad_px)) * sy

    # Draw GPS flight path line
    gps_pts = [to_canvas(r['gps_lat'], r['gps_lon']) for r in rows]
    for i in range(len(gps_pts)-1):
        draw.line([gps_pts[i], gps_pts[i+1]], fill="lime", width=3)

    # Draw each point with colour by status
    COLOURS = {
        "LOCK":         "#00FFFF",   # cyan
        "EKF_REJECTED": "red",
        "LOW_CONF":     "orange",
    }

    for i, r in enumerate(rows):
        gx, gy = gps_pts[i]
        tx, ty = to_canvas(r['trn_lat'], r['trn_lon'])
        col = COLOURS.get(r['status'], "white")

        if r['status'] == "LOCK":
            draw.line([gx, gy, tx, ty], fill="white", width=1)

        draw.ellipse([gx-6, gy-6, gx+6, gy+6], fill="lime")
        draw.ellipse([tx-5, ty-5, tx+5, ty+5], fill=col)

        label = f"{r['photo'][:8]} {r['err_m']:.0f}m"
        draw.text((gx+7, gy-10), label, fill="white",
                  stroke_width=2, stroke_fill="black")

    # Legend
    draw.rectangle([5, 5, 280, 95], fill=(0, 0, 0, 200))
    draw.ellipse([12,14,22,24], fill="lime");    draw.text((28,12), "GPS ground truth",  fill="lime")
    draw.ellipse([12,34,22,44], fill="#00FFFF"); draw.text((28,32), "TRN LOCK (accepted)", fill="#00FFFF")
    draw.ellipse([12,54,22,64], fill="red");     draw.text((28,52), "EKF REJECTED",      fill="red")
    draw.ellipse([12,74,22,84], fill="orange");  draw.text((28,72), "Low confidence",    fill="orange")

    im.save(OUT_MAP)
    print(f"\n[MAP] Saved to {OUT_MAP}")

    accepted = sum(1 for r in rows if r['status'] == "LOCK")
    rejected = sum(1 for r in rows if r['status'] == "EKF_REJECTED")
    low_conf = sum(1 for r in rows if r['status'] == "LOW_CONF")
    print(f"\n{'='*55}")
    print(f"  EKF SIMULATION SUMMARY  ({len(rows)} frames)")
    print(f"{'='*55}")
    print(f"  Accepted (LOCK)    : {accepted}")
    print(f"  EKF Rejected       : {rejected}")
    print(f"  Low Confidence     : {low_conf}")
    print(f"  EKF Reject Rate    : {rejected/(accepted+rejected)*100:.1f}%  "
          f"(of high-confidence frames)")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
