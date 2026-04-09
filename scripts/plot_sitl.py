"""
plot_sitl.py  –  Flight path visualiser for video_flight_history.json / sitl_flight_history.json

Visual design:
  - Orthophoto background with a zoomed inset on the busiest area
  - TRN path drawn as a continuous gradient line (green→yellow→red by error magnitude)
  - GPS path drawn smoothed and dashed in white  — the raw GPS is low-resolution (4dp ≈ 11m)
    so it appears staircase-like; smoothing makes the drone's actual trajectory clear
  - Anchor locks drawn as cyan rings
  - Stats panel showing per-leg accuracy
"""

import json, math, os
import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

ORTHO_PATH = r"C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif"
ORTHO_RES  = 0.5
OUT_MAP    = "sitl_ekf_simulation_map.png"
SIZE       = 1400          # output image size in pixels
GPS_SMOOTH = 15            # moving-average window for GPS path display (hides quantization)


# ── Helpers ──────────────────────────────────────────────────────────────────
def smooth_path(pts, w):
    """Apply a simple moving-average to a list of (x,y) tuples."""
    out = []
    for i in range(len(pts)):
        lo, hi = max(0, i - w//2), min(len(pts), i + w//2 + 1)
        xs = [pts[j][0] for j in range(lo, hi)]
        ys = [pts[j][1] for j in range(lo, hi)]
        out.append((sum(xs)/len(xs), sum(ys)/len(ys)))
    return out

def err_to_colour(err_m, max_err=80):
    """Map error in metres to an RGB colour: green→yellow→red."""
    t = min(err_m / max_err, 1.0)
    if t < 0.5:
        r = int(255 * (t * 2))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (t - 0.5) * 2))
    return (r, g, 0)

def draw_thick_line(draw, p1, p2, colour, width=4):
    draw.line([p1, p2], fill=colour, width=width)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    src_file = "sitl_flight_history.json" if os.path.exists("sitl_flight_history.json") \
               else "video_flight_history.json"
    if not os.path.exists(src_file):
        print("No flight history JSON found."); return

    with open(src_file) as f:
        rows = json.load(f)
    if not rows:
        print("Empty history."); return

    src    = rasterio.open(ORTHO_PATH)
    to_mga = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)

    # Centre the map on the GPS centroid with generous padding
    all_lats = [r['gps_lat'] for r in rows]
    all_lons = [r['gps_lon'] for r in rows]
    cen_lat  = (min(all_lats) + max(all_lats)) / 2
    cen_lon  = (min(all_lons) + max(all_lons)) / 2
    pad_m    = 350
    pad_px   = int(pad_m / ORTHO_RES)

    cen_e, cen_n = to_mga.transform(cen_lon, cen_lat)
    cen_col, cen_row = ~src.transform * (cen_e, cen_n)
    cen_col, cen_row = int(cen_col), int(cen_row)

    bg_win = rasterio.windows.Window(cen_col-pad_px, cen_row-pad_px, pad_px*2, pad_px*2)
    bg     = src.read(window=bg_win)
    bg_rgb = cv2.cvtColor(np.transpose(bg, (1,2,0)), cv2.COLOR_RGBA2RGB)

    # Slightly darken background so overlays pop
    bg_rgb = (bg_rgb * 0.65).astype(np.uint8)

    im   = Image.fromarray(bg_rgb).resize((SIZE, SIZE), Image.LANCZOS)
    draw = ImageDraw.Draw(im, "RGBA")
    sx   = SIZE / (pad_px * 2)
    sy   = SIZE / (pad_px * 2)

    def to_canvas(lat, lon):
        e, n = to_mga.transform(lon, lat)
        col, row = ~src.transform * (e, n)
        return ((col - (cen_col - pad_px)) * sx,
                (row - (cen_row - pad_px)) * sy)

    # Pre-compute canvas points
    gps_pts = [to_canvas(r['gps_lat'], r['gps_lon']) for r in rows]
    trn_pts = [to_canvas(r['trn_lat'], r['trn_lon']) for r in rows]
    errs    = [r['err_m'] for r in rows]

    # ── 1. GPS path — smoothed dashed line ───────────────────────────────────
    gps_smooth = smooth_path(gps_pts, GPS_SMOOTH)
    for i in range(0, len(gps_smooth)-1, 2):   # every other segment = dashed effect
        draw.line([gps_smooth[i], gps_smooth[i+1]],
                  fill=(255, 255, 255, 140), width=2)

    # ── 2. TRN path — solid gradient line coloured by error ──────────────────
    for i in range(len(trn_pts)-1):
        col = err_to_colour(errs[i]) + (220,)   # add alpha
        draw.line([trn_pts[i], trn_pts[i+1]], fill=col, width=4)

    # ── 3. Anchor lock rings ─────────────────────────────────────────────────
    for i, r in enumerate(rows):
        if r['status'] == 'LOCK':
            tx, ty = trn_pts[i]
            draw.ellipse([tx-12, ty-12, tx+12, ty+12],
                         outline=(0, 255, 255, 255), width=2)

    # ── 4. Error tick marks every 5 seconds (every ~150 frames @ 30fps) ──────
    fps_approx = 30
    tick_every = fps_approx * 5
    for i in range(0, len(rows), tick_every):
        tx, ty = trn_pts[i]
        col    = err_to_colour(errs[i])
        t_sec  = i / fps_approx
        label  = f"{errs[i]:.0f}m"
        # Dot at TRN position
        draw.ellipse([tx-6, ty-6, tx+6, ty+6], fill=col+(255,), outline=(0,0,0,180), width=1)
        # Time + error label
        draw.text((tx+9, ty-9), f"t={t_sec:.0f}s\n{label}",
                  fill=(255, 255, 255, 220), stroke_width=1)

    # ── 5. Start / End markers ────────────────────────────────────────────────
    sx0, sy0 = trn_pts[0]
    ex0, ey0 = trn_pts[-1]
    draw.ellipse([sx0-10, sy0-10, sx0+10, sy0+10], fill=(50,255,50,255), outline=(0,0,0,200), width=2)
    draw.text((sx0+13, sy0-10), "START", fill=(50,255,50,255), stroke_width=1)
    draw.ellipse([ex0-10, ey0-10, ex0+10, ey0+10], fill=(255,80,80,255), outline=(0,0,0,200), width=2)
    draw.text((ex0+13, ey0-10), "END",   fill=(255,80,80,255), stroke_width=1)

    # ── 6. Per-leg stats ─────────────────────────────────────────────────────
    outbound  = [r for r in rows if r['photo'] <= 'f_02459']
    uturn     = [r for r in rows if 'f_02460' <= r['photo'] <= 'f_03179']
    returnleg = [r for r in rows if r['photo'] > 'f_03179']

    def leg_avg(seg):
        return sum(r['err_m'] for r in seg)/len(seg) if seg else 0
    def leg_p90(seg):
        s = sorted(r['err_m'] for r in seg)
        return s[int(len(s)*0.9)] if seg else 0

    stats_lines = [
        ("TERCOM Video TRN — Full Flight Analysis", (255,255,255)),
        ("", None),
        (f"  Outbound (0–82s)    avg {leg_avg(outbound):.1f}m   P90 {leg_p90(outbound):.1f}m",  (100,255,100)),
        (f"  U-turn  (82–106s)   avg {leg_avg(uturn):.1f}m    P90 {leg_p90(uturn):.1f}m",       (255,220,60)),
        (f"  Return  (106–173s)  avg {leg_avg(returnleg):.1f}m   P90 {leg_p90(returnleg):.1f}m", (255,130,60)),
        (f"  FULL FLIGHT         avg {leg_avg(rows):.1f}m   P90 {leg_p90(rows):.1f}m",           (255,255,255)),
        ("", None),
        (f"  Frames: {len(rows):,}   Anchors: {sum(1 for r in rows if r['status']=='LOCK')}", (200,200,200)),
    ]

    panel_h = len(stats_lines) * 22 + 20
    draw.rectangle([8, 8, 420, 8 + panel_h], fill=(0,0,0,185))
    y = 16
    for txt, col in stats_lines:
        if txt and col:
            draw.text((16, y), txt, fill=col + (255,), stroke_width=1)
        y += 22

    # ── 7. Legend ─────────────────────────────────────────────────────────────
    lx, ly = 8, SIZE - 155
    draw.rectangle([lx, ly, lx+310, ly+145], fill=(0,0,0,175))
    items = [
        ((255,255,255,140), "──── GPS path (smoothed, dashed)", False),
        ((100,255,0,220),   "──── TRN < 15m error",             False),
        ((255,220,0,220),   "──── TRN 15–50m error",            False),
        ((255,50,50,220),   "──── TRN > 50m error",             False),
        ((0,255,255,255),   "○   Structural anchor LOCK",       True),
    ]
    ly2 = ly + 12
    for col, label, is_ring in items:
        if is_ring:
            draw.ellipse([lx+12, ly2, lx+24, ly2+12], outline=col, width=2)
        else:
            draw.line([(lx+8, ly2+6), (lx+28, ly2+6)], fill=col, width=3)
        draw.text((lx+34, ly2), label, fill=(230,230,230,255))
        ly2 += 24
    note = "Note: raw GPS has ~11m grid resolution (4 d.p. SRT)"
    draw.text((lx+8, ly2+4), note, fill=(180,180,180,200))

    im.save(OUT_MAP)
    print(f"\n[MAP] Saved to {OUT_MAP}")

if __name__ == "__main__":
    main()
