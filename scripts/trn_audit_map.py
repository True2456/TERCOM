"""
Generate audit map and summary from trn_vs_gps_results.csv
"""
import csv, json
import rasterio
import numpy as np
import cv2
from pyproj import Transformer
from PIL import Image, ImageDraw

DEM_PATH   = "5m_DEM.tif"
CSV_PATH   = "trn_vs_gps_results.csv"
OUT_MAP    = "trn_vs_gps_map.png"
PAD_M      = 400
DEM_RES    = 5.0

rows = list(csv.DictReader(open(CSV_PATH)))
for r in rows:
    r["error_m"] = float(r["error_m"])
    r["psr"]     = float(r["psr"])
    r["gps_lat"] = float(r["gps_lat"])
    r["gps_lon"] = float(r["gps_lon"])
    r["trn_lat"] = float(r["trn_lat"])
    r["trn_lon"] = float(r["trn_lon"])

all_lats = [r["gps_lat"] for r in rows]
all_lons = [r["gps_lon"] for r in rows]
cen_lat  = sum(all_lats) / len(all_lats)
cen_lon  = sum(all_lons) / len(all_lons)

with rasterio.open(DEM_PATH) as src:
    to_mga   = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

    cen_e, cen_n = to_mga.transform(cen_lon, cen_lat)
    pad_px = int(PAD_M / DEM_RES)
    cen_col, cen_row = ~src.transform * (cen_e, cen_n)
    cen_col, cen_row = int(cen_col), int(cen_row)

    win = rasterio.windows.Window(cen_col - pad_px, cen_row - pad_px, pad_px*2, pad_px*2)
    dem = src.read(1, window=win)
    dem[dem < -100] = 0
    hs = cv2.normalize(dem, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hs_rgb = cv2.cvtColor(hs, cv2.COLOR_GRAY2RGB)
    SIZE = 1024
    im = Image.fromarray(hs_rgb).resize((SIZE, SIZE), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    sx = SIZE / (pad_px * 2)
    sy = SIZE / (pad_px * 2)

    def world_to_px(lat, lon):
        e, n = to_mga.transform(lon, lat)
        col, row = ~src.transform * (e, n)
        x = (col - (cen_col - pad_px)) * sx
        y = (row - (cen_row - pad_px)) * sy
        return x, y

    # Draw GPS track line
    gps_pts = [world_to_px(r["gps_lat"], r["gps_lon"]) for r in rows]
    for i in range(len(gps_pts) - 1):
        draw.line([gps_pts[i], gps_pts[i+1]], fill="lime", width=2)

    # Draw TRN track line
    trn_pts = [world_to_px(r["trn_lat"], r["trn_lon"]) for r in rows]
    for i in range(len(trn_pts) - 1):
        draw.line([trn_pts[i], trn_pts[i+1]], fill="cyan", width=2)

    for i, r in enumerate(rows):
        gx, gy = gps_pts[i]
        tx, ty = trn_pts[i]

        # Error line (red)
        draw.line([gx, gy, tx, ty], fill="red", width=1)

        # GPS dot (green)
        draw.ellipse([gx-5, gy-5, gx+5, gy+5], fill="lime")

        # TRN dot (cyan)
        draw.ellipse([tx-4, ty-4, tx+4, ty+4], fill="cyan")

        # Label every 3rd photo to avoid clutter
        if i % 3 == 0:
            draw.text((gx+7, gy-10), f"{r['photo'][:8]} {r['error_m']:.0f}m", fill="white")

    # Legend
    draw.rectangle([5, 5, 220, 65], fill=(0, 0, 0))
    draw.ellipse([12, 14, 22, 24], fill="lime");  draw.text((26, 12), "GPS Truth", fill="lime")
    draw.ellipse([12, 34, 22, 44], fill="cyan");  draw.text((26, 32), "TRN Estimate", fill="cyan")
    draw.text((12, 50), "Red lines = error vectors", fill="red")

    # Scale bar (100m)
    bar_px = int(100 / DEM_RES * sx)
    draw.rectangle([SIZE-bar_px-10, SIZE-25, SIZE-10, SIZE-15], fill="white")
    draw.text((SIZE-bar_px-10, SIZE-40), "100m", fill="white")

    im.save(OUT_MAP)
    print(f"Map saved: {OUT_MAP}")

# Summary
errors = [r["error_m"] for r in rows]
psrs   = [r["psr"] for r in rows]
print(f"\n{'='*50}")
print(f"  TRN vs GPS ACCURACY SUMMARY  ({len(rows)} photos)")
print(f"{'='*50}")
print(f"  Mean error    : {sum(errors)/len(errors):.1f} m")
print(f"  Min error     : {min(errors):.1f} m  ({rows[errors.index(min(errors))]['photo']})")
print(f"  Max error     : {max(errors):.1f} m  ({rows[errors.index(max(errors))]['photo']})")
print(f"  Mean PSR      : {sum(psrs)/len(psrs):.2f}")
print(f"  Tile footprint: {rows[0]['tile_footprint_m']} m  (150m AGL, DJI Mini 3)")
print(f"{'='*50}")
print(f"\nPer-photo breakdown:")
print(f"{'Photo':<16} {'GPS Lat':>12} {'GPS Lon':>13} {'TRN Lat':>12} {'TRN Lon':>13} {'Err(m)':>8} {'PSR':>6}")
print("-" * 85)
for r in rows:
    print(f"{r['photo']:<16} {r['gps_lat']:>12.7f} {r['gps_lon']:>13.7f} "
          f"{r['trn_lat']:>12.7f} {r['trn_lon']:>13.7f} {r['error_m']:>8.1f} {r['psr']:>6.2f}")
