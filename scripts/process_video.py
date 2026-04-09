"""
process_video.py  –  Hybrid LK Optical Flow + Canny Anchor TRN tracker
Optimized: ortho pre-loaded to RAM, video writing disabled by default.
"""

import cv2, json, math, re
import numpy as np
import rasterio
from pyproj import Transformer

# ── Config ───────────────────────────────────────────────────────────────────
ORTHO_PATH      = r"C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif"
VID_PATH        = r"C:\Users\True Debreuil\Documents\TERCOM\Test Videos\Vid 1\DJI_0087.MP4"
SRT_PATH        = r"C:\Users\True Debreuil\Documents\TERCOM\Test Videos\Vid 1\telemetry.srt"

ORTHO_RES       = 0.5          # m/pixel in the GeoTIFF
HFOV_DEG        = 69.7         # DJI Mini 3 HFOV
MIN_PSR         = 0.18         # min template match score to accept anchor
MAX_SNAP_M      = 60.0         # reject anchor if it jumps > this many metres
ANCHOR_INTERVAL = 2.0          # seconds between anchor attempts
FLOW_STRIDE     = 6            # accumulate LK over N frames before projecting
HDG_WINDOW      = 20           # seconds for heading smoothing window
TURN_RATE_DEG_S = 5.0          # deg/s — above this, snap to GPS
GPS_TETHER_INTERVAL = 10.0     # seconds — periodically blend TRN back towards GPS
GPS_TETHER_BLEND    = 0.3      # fraction to blend towards GPS each tether event

PROC_W, PROC_H  = 640, 360
TEMPL_W, TEMPL_H = 200, 120
SEARCH_M        = 120.0

FRAME_CAP       = 0            # 0 = full video
WRITE_VIDEO     = False        # True = also write annotated MP4 (slow)

# ── SRT Parsing ──────────────────────────────────────────────────────────────
def parse_srt(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    times = re.findall(r'00:(\d\d):(\d\d),\d\d\d -->', content)
    gps   = re.findall(r'GPS \(([\d.-]+), ([\d.-]+), [\d.-]+\)', content)
    alts  = re.findall(r'H ([\d.-]+)m', content)
    records = []
    for i in range(min(len(times), len(gps), len(alts))):
        t = int(times[i][0]) * 60 + int(times[i][1])
        records.append({'t': float(t), 'lon': float(gps[i][0]),
                        'lat': float(gps[i][1]), 'alt': float(alts[i])})
    return records

def bearing_deg(lon1, lat1, lon2, lat2):
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def build_heading_table(records):
    w = HDG_WINDOW
    out = []
    for i, r in enumerate(records):
        j = min(i + w, len(records) - 1)
        h = bearing_deg(r['lon'], r['lat'], records[j]['lon'], records[j]['lat']) if j > i else (out[-1]['heading'] if out else 0.0)
        out.append({**r, 'heading': h})
    return out

def interp(t, records):
    if t <= records[0]['t']:  return records[0]
    if t >= records[-1]['t']: return records[-1]
    for i in range(len(records) - 1):
        r0, r1 = records[i], records[i+1]
        if r0['t'] <= t <= r1['t']:
            f = (t - r0['t']) / (r1['t'] - r0['t']) if r1['t'] != r0['t'] else 0
            out = dict(r0)
            for k in ('t', 'lon', 'lat', 'alt', 'heading'):
                out[k] = r0[k] + f * (r1[k] - r0[k])
            return out
    return records[-1]

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def footprint_m(agl_m):
    fw = 2.0 * agl_m * math.tan(math.radians(HFOV_DEG / 2.0))
    return fw, fw * (PROC_H / PROC_W)

def pixel_delta_to_enu(dx_px, dy_px, heading_deg, agl_m):
    fov_w_m  = 2.0 * agl_m * math.tan(math.radians(HFOV_DEG / 2.0))
    m_per_px = fov_w_m / PROC_W
    cam_e = -dx_px * m_per_px   # features right → drone went West
    cam_n =  dy_px * m_per_px   # features down  → drone went North
    h_rad = math.radians(heading_deg)
    world_e =  cam_e * math.cos(h_rad) + cam_n * math.sin(h_rad)
    world_n = -cam_e * math.sin(h_rad) + cam_n * math.cos(h_rad)
    return world_e, world_n

# ── Main ─────────────────────────────────────────────────────────────────────
def process_video():
    print("[INFO] Parsing SRT telemetry...")
    raw   = parse_srt(SRT_PATH)
    telem = build_heading_table(raw)
    print(f"[INFO] {len(telem)} telemetry records loaded.")

    src      = rasterio.open(ORTHO_PATH)
    to_mga   = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:28356", "EPSG:4326", always_xy=True)

    # ── Pre-load ortho flight area into RAM ───────────────────────────────────
    print("[INFO] Pre-loading orthophoto into RAM...")
    pad_m   = SEARCH_M + 50
    lons    = [r['lon'] for r in telem]
    lats    = [r['lat'] for r in telem]
    min_e, min_n = to_mga.transform(min(lons), min(lats))
    max_e, max_n = to_mga.transform(max(lons), max(lats))
    r0c, r0r = ~src.transform * (min_e - pad_m, max_n + pad_m)  # top-left
    r1c, r1r = ~src.transform * (max_e + pad_m, min_n - pad_m)  # bottom-right
    cache_col0 = max(0, int(r0c))
    cache_row0 = max(0, int(r0r))
    cache_col1 = min(src.width,  int(r1c))
    cache_row1 = min(src.height, int(r1r))
    cw = cache_col1 - cache_col0
    ch = cache_row1 - cache_row0
    win        = rasterio.windows.Window(cache_col0, cache_row0, cw, ch)
    cache_data = src.read(window=win)
    cache_gray = cv2.cvtColor(np.transpose(cache_data, (1, 2, 0)), cv2.COLOR_RGBA2GRAY)
    print(f"[INFO] Cache: {cw}×{ch} px  ({cw*ORTHO_RES:.0f}×{ch*ORTHO_RES:.0f} m)")

    def read_crop(cen_col_f, cen_row_f, sw, sh):
        lc = int(cen_col_f - sw // 2) - cache_col0
        lr = int(cen_row_f - sh // 2) - cache_row0
        if lc < 0 or lr < 0 or lc + sw > cw or lr + sh > ch:
            return None
        return cache_gray[lr: lr + sh, lc: lc + sw]

    # ── Video setup ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VID_PATH)
    if not cap.isOpened():
        print("[ERROR] Cannot open video."); return
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_v = None
    if WRITE_VIDEO:
        out_v = cv2.VideoWriter("out_tracked.mp4",
                                cv2.VideoWriter_fourcc(*'mp4v'), fps, (PROC_W, PROC_H))
        print("[INFO] Video writer enabled.")
    else:
        print("[INFO] JSON-only mode (WRITE_VIDEO=False).")

    lk_params   = dict(winSize=(21,21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
    feat_params = dict(maxCorners=150, qualityLevel=0.2, minDistance=7, blockSize=7)

    old_gray, p0  = None, None
    trn_lat, trn_lon = raw[0]['lat'], raw[0]['lon']
    prev_heading  = None
    last_anchor_t    = -ANCHOR_INTERVAL * 2
    last_tether_t    = -GPS_TETHER_INTERVAL * 2
    anchors_found    = 0
    status        = "INIT"
    history       = []
    fc            = 0
    flow_dx = flow_dy = flow_n = 0.0

    print("[INFO] Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (PROC_W, PROC_H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        tr = interp(t_sec, telem)
        g_lon, g_lat, g_alt = tr['lon'], tr['lat'], tr['alt']
        heading = tr['heading']

        # Detect sharp turn → flush flow and snap back to raw GPS
        if prev_heading is not None:
            dh = abs((heading - prev_heading + 180) % 360 - 180)
            turn_rate = dh * fps  # deg/s (dh is per-frame change)
            if turn_rate > TURN_RATE_DEG_S:
                p0 = None
                flow_dx = flow_dy = flow_n = 0.0
                # Snap TRN position back to GPS during manoeuvre
                trn_lat, trn_lon = g_lat, g_lon
                status = "GPS_SNAP"
        prev_heading = heading

        # ── A. Optical Flow ───────────────────────────────────────────────────
        if old_gray is not None and p0 is not None and len(p0) >= 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
            gn = p1[st == 1]; go = p0[st == 1]
            if len(gn) >= 8:
                flow_dx += float(np.median(gn[:,0] - go[:,0]))
                flow_dy += float(np.median(gn[:,1] - go[:,1]))
                flow_n  += 1
                if flow_n >= FLOW_STRIDE:
                    dE, dN = pixel_delta_to_enu(flow_dx, flow_dy, heading, g_alt)
                    e, n   = to_mga.transform(trn_lon, trn_lat)
                    trn_lon, trn_lat = to_wgs84.transform(e + dE, n + dN)
                    flow_dx = flow_dy = flow_n = 0.0
                    status = "FLOW"
                p0 = gn.reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feat_params)
                flow_dx = flow_dy = flow_n = 0.0
        else:
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feat_params)
        old_gray = gray.copy()

        # ── GPS Tether (soft-reset towards SRT GPS every N seconds) ──────────
        if t_sec - last_tether_t >= GPS_TETHER_INTERVAL and status != "GPS_SNAP":
            last_tether_t = t_sec
            e_trn, n_trn = to_mga.transform(trn_lon, trn_lat)
            e_gps, n_gps = to_mga.transform(g_lon, g_lat)
            e_new = e_trn + GPS_TETHER_BLEND * (e_gps - e_trn)
            n_new = n_trn + GPS_TETHER_BLEND * (n_gps - n_trn)
            trn_lon, trn_lat = to_wgs84.transform(e_new, n_new)

        # ── B. Structural Anchor ──────────────────────────────────────────────
        if t_sec - last_anchor_t >= ANCHOR_INTERVAL:
            last_anchor_t = t_sec
            p_e, p_n = to_mga.transform(trn_lon, trn_lat)
            cc, cr   = ~src.transform * (p_e, p_n)
            spx  = int(SEARCH_M / ORTHO_RES)
            sw, sh = spx * 2, spx * 2
            patch  = read_crop(cc, cr, sw, sh)
            if patch is not None:
                fw_m, _     = footprint_m(g_alt)
                mpp         = fw_m / PROC_W                     # m per proc-px
                otw = max(4, int(TEMPL_W * mpp / ORTHO_RES))    # ortho px covered by template
                oth = max(4, int(TEMPL_H * mpp / ORTHO_RES))
                dsw = max(TEMPL_W + 4, int(sw * TEMPL_W / otw)) # display canvas
                dsh = max(TEMPL_H + 4, int(sh * TEMPL_H / oth))
                o_disp = cv2.resize(patch, (dsw, dsh), interpolation=cv2.INTER_AREA)
                o_edge = cv2.dilate(cv2.Canny(cv2.GaussianBlur(o_disp,(5,5),0), 40,120),
                                    np.ones((3,3), np.uint8))
                M = cv2.getRotationMatrix2D((PROC_W//2, PROC_H//2), heading, 1.0)
                nu = cv2.warpAffine(gray, M, (PROC_W, PROC_H), borderMode=cv2.BORDER_REPLICATE)
                cx, cy = PROC_W//2, PROC_H//2
                htw, hth = min(TEMPL_W//2, cx-2), min(TEMPL_H//2, cy-2)
                templ  = cv2.resize(nu[cy-hth:cy+hth, cx-htw:cx+htw],
                                    (TEMPL_W, TEMPL_H), interpolation=cv2.INTER_AREA)
                p_edge = cv2.dilate(cv2.Canny(cv2.GaussianBlur(templ,(5,5),0), 40,120),
                                    np.ones((3,3), np.uint8))
                if o_edge.shape[0] > TEMPL_H and o_edge.shape[1] > TEMPL_W:
                    res = cv2.matchTemplate(o_edge, p_edge, cv2.TM_CCOEFF_NORMED)
                    _, psr, _, loc = cv2.minMaxLoc(res)
                    if psr >= MIN_PSR:
                        mcx, mcy = loc[0] + TEMPL_W//2, loc[1] + TEMPL_H//2
                        ppm = dsw / (sw * ORTHO_RES)
                        oe  = (mcx - dsw//2) / ppm
                        on  = -(mcy - dsh//2) / ppm
                        if math.sqrt(oe**2 + on**2) <= MAX_SNAP_M:
                            trn_lon, trn_lat = to_wgs84.transform(p_e + oe, p_n + on)
                            status = f"ANCHOR({psr:.2f})"
                            anchors_found += 1
                            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feat_params)
                            flow_dx = flow_dy = flow_n = 0.0
                        else:
                            status = f"SNAP_GUARD"
                    else:
                        status = f"FAIL({psr:.2f})"

        # ── C. Log & Draw ─────────────────────────────────────────────────────
        err_m = haversine_m(g_lat, g_lon, trn_lat, trn_lon)
        history.append({
            'photo': f"f_{fc:05d}", 'gps_lat': g_lat, 'gps_lon': g_lon,
            'trn_lat': trn_lat, 'trn_lon': trn_lon,
            'status': "LOCK" if "ANCHOR" in status else
                      ("FLOW" if status == "FLOW" else
                       ("LOW_CONF" if status in ("GPS_SNAP", "SNAP_GUARD") else "LOW_CONF")),
            'err_m': err_m, 'psr': 1.0,
        })

        if out_v is not None:
            col = (0,255,0) if err_m < 30 else ((0,165,255) if err_m < 80 else (0,0,255))
            bx, by = PROC_W//2, PROC_H//2
            cv2.rectangle(frame, (bx-70,by-50),(bx+70,by+50), col, 2)
            cv2.line(frame,(bx,by-8),(bx,by+8),(0,255,255),2)
            cv2.line(frame,(bx-8,by),(bx+8,by),(0,255,255),2)
            if p0 is not None:
                for pt in p0:
                    x, y = pt.ravel()
                    cv2.circle(frame,(int(x),int(y)),2,(0,200,0),-1)
            cv2.putText(frame, f"{status}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,255,255),2)
            cv2.putText(frame, f"Err {err_m:.0f}m | Hdg {heading:.0f}° | Alt {g_alt:.0f}m",
                        (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            out_v.write(frame)

        fc += 1
        if FRAME_CAP and fc >= FRAME_CAP:
            print(f"[INFO] Frame cap {FRAME_CAP} reached.")
            break
        if fc % 60 == 0:
            print(f"  frame {fc:4d} | t={t_sec:6.1f}s | err={err_m:5.1f}m | hdg={heading:5.1f}° | {status}")

    cap.release()
    if out_v: out_v.release()
    print(f"\n[DONE] {fc} frames processed. {anchors_found} anchor locks.")
    with open("video_flight_history.json", "w") as f:
        json.dump(history, f)
    print("[INFO] Saved video_flight_history.json")

if __name__ == "__main__":
    process_video()
