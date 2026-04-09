import os, json, math
import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

ORTHO_PATH = r"C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif"
ORTHO_RES  = 0.5

def main():
    if not os.path.exists("sitl_flight_history.json"):
        print("No sitl_flight_history.json found! Need to run the SITL flight first.")
        return
        
    with open("sitl_flight_history.json") as f:
        rows = json.load(f)
        
    if not rows:
        print("Empty history.")
        return

    src = rasterio.open(ORTHO_PATH)
    to_mga = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)

    all_lats = [r['gps_lat'] for r in rows]
    all_lons = [r['gps_lon'] for r in rows]
    cen_lat  = (min(all_lats) + max(all_lats)) / 2
    cen_lon  = (min(all_lons) + max(all_lons)) / 2
    pad_m    = 300
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

    COLOURS = {
        "LOCK":         "#00FFFF",   # cyan  – structural anchor confirmed
        "FLOW":         "#FFD700",   # gold  – optical flow tracking
        "EKF_REJECTED": "#FF4444",   # red   – EKF kinematic gate rejection
        "LOW_CONF":     "#FF8C00",   # orange – low template confidence
    }

    for i, r in enumerate(rows):
        gx, gy = gps_pts[i]
        tx, ty = to_canvas(r['trn_lat'], r['trn_lon'])
        col = COLOURS.get(r['status'], "#AAAAAA")

        # Draw connecting line from GPS truth to TRN estimate for LOCKs
        if r['status'] == "LOCK":
            draw.line([gx, gy, tx, ty], fill="white", width=1)

        # GPS ground truth dot (always drawn)
        draw.ellipse([gx-5, gy-5, gx+5, gy+5], fill="lime")

        # TRN estimate dot (always drawn except LOW_CONF – not reliable enough)
        if r['status'] != "LOW_CONF":
            draw.ellipse([tx-4, ty-4, tx+4, ty+4], fill=col)

        # Error label every 5th frame for readability
        if i % 5 == 0 and r['status'] in ("LOCK", "FLOW"):
            label = f"{r['err_m']:.0f}m"
            draw.text((tx+6, ty-9), label, fill="white", stroke_width=1)

    # Legend
    draw.rectangle([5, 5, 290, 115], fill=(0, 0, 0, 180))
    draw.ellipse([12,14,22,24], fill="lime");     draw.text((28, 12), "GPS ground truth",          fill="lime")
    draw.ellipse([12,34,22,44], fill="#FFD700");  draw.text((28, 32), "Optical flow tracking",     fill="#FFD700")
    draw.ellipse([12,54,22,64], fill="#00FFFF");  draw.text((28, 52), "Structural anchor LOCK",    fill="#00FFFF")
    draw.ellipse([12,74,22,84], fill="#FF4444");  draw.text((28, 72), "EKF rejected",              fill="#FF4444")
    draw.ellipse([12,94,22,104], fill="#FF8C00"); draw.text((28, 92), "Low confidence (skipped)",  fill="#FF8C00")

    OUT_MAP = "sitl_ekf_simulation_map.png"
    im.save(OUT_MAP)
    print(f"\n[MAP] Saved SITL representation to {OUT_MAP}")

if __name__ == "__main__":
    main()
