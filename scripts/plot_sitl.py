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
        
        if r['status'] != "LOW_CONF":
            draw.ellipse([tx-5, ty-5, tx+5, ty+5], fill=col)

            # Draw every 3rd label so it doesn't get too messy
            if r['status'] == "LOCK" and i % 3 == 0:
                label = f"{r['err_m']:.0f}m"
                draw.text((tx+7, ty-10), label, fill="white", stroke_width=2, stroke_stroke="black")

    # Legend
    draw.rectangle([5, 5, 280, 95], fill=(0, 0, 0, 200))
    draw.ellipse([12,14,22,24], fill="lime");    draw.text((28,12), "GPS ground truth",  fill="lime")
    draw.ellipse([12,34,22,44], fill="#00FFFF"); draw.text((28,32), "TRN LOCK (accepted)", fill="#00FFFF")
    draw.ellipse([12,54,22,64], fill="red");     draw.text((28,52), "EKF REJECTED",      fill="red")
    draw.ellipse([12,74,22,84], fill="orange");  draw.text((28,72), "Low confidence",    fill="orange")

    OUT_MAP = "sitl_ekf_simulation_map.png"
    im.save(OUT_MAP)
    print(f"\n[MAP] Saved SITL representation to {OUT_MAP}")

if __name__ == "__main__":
    main()
