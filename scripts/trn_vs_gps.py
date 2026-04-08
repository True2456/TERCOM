"""
TRN vs GPS Comparison Pipeline
- Reads JPGs from 'Test photos 2' (READ-ONLY, EXIF never modified)
- Computes ground footprint at 150m AGL using camera FOV
- Matches photo against local DEM tile via phase correlation (edge mode)
- Reports TRN estimate vs GPS ground truth with error distance (m)
- Saves results to CSV + visual audit map
"""

import os, json, csv, math
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image, ImageDraw, ImageFont
from navigation.matcher import FFTMatcher

# ── Config ──────────────────────────────────────────────────────────────────
PHOTO_DIR      = "Test photos 2"
DEM_PATH       = "5m_DEM.tif"
GPS_TRUTH_FILE = "gps_ground_truth.json"
OUT_CSV        = "trn_vs_gps_results.csv"
OUT_MAP        = "trn_vs_gps_map.png"
DEM_RESOLUTION = 5.0        # metres per pixel
AGL_M          = 150.0      # True altitude above ground level
HFOV_DEG       = 82.1       # DJI Mini 3 horizontal field of view (degrees)
SEARCH_PAD_M   = 300        # Search ±300m around GPS truth centroid
STEP_PX        = 5          # Sliding window step (DEM pixels)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def compute_footprint_px(agl, hfov_deg, img_w, img_h, dem_res):
    """Returns (tile_w_px, tile_h_px) in DEM pixels for given AGL and FOV."""
    gnd_w = 2 * agl * math.tan(math.radians(hfov_deg / 2))
    aspect = img_h / img_w
    gnd_h = gnd_w * aspect
    return max(1, int(gnd_w / dem_res)), max(1, int(gnd_h / dem_res))

def main():
    matcher = FFTMatcher()

    with open(GPS_TRUTH_FILE) as f:
        gps_truth = json.load(f)

    jpgs = sorted([k for k in gps_truth if k.endswith(".JPG")])
    print(f"Processing {len(jpgs)} photos with GPS ground truth\n")
    print(f"AGL: {AGL_M}m | HFOV: {HFOV_DEG}° | Search pad: ±{SEARCH_PAD_M}m\n")

    # Centroid of all GPS points → search anchor
    all_lats = [gps_truth[j]["lat"] for j in jpgs]
    all_lons = [gps_truth[j]["lon"] for j in jpgs]
    cen_lat = sum(all_lats) / len(all_lats)
    cen_lon = sum(all_lons) / len(all_lons)
    print(f"Mission centroid: {cen_lat:.7f}, {cen_lon:.7f}\n")

    rows = []

    with rasterio.open(DEM_PATH) as src:
        to_mga   = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        # Build search window around centroid
        cen_e, cen_n = to_mga.transform(cen_lon, cen_lat)
        pad_px = int(SEARCH_PAD_M / DEM_RESOLUTION)
        cen_col, cen_row = ~src.transform * (cen_e, cen_n)
        cen_col, cen_row = int(cen_col), int(cen_row)

        win = rasterio.windows.Window(
            cen_col - pad_px, cen_row - pad_px,
            pad_px * 2, pad_px * 2
        )
        dem_patch = src.read(1, window=win)
        dem_patch[dem_patch < -100] = 0

        # Normalise DEM for display / matching
        hs = cv2.normalize(dem_patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        for photo_name in jpgs:
            truth = gps_truth[photo_name]
            t_lat, t_lon = truth["lat"], truth["lon"]

            photo_path = os.path.join(PHOTO_DIR, photo_name)
            img_gray = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"  {photo_name}: COULD NOT READ")
                continue

            img_h, img_w = img_gray.shape
            tile_w, tile_h = compute_footprint_px(AGL_M, HFOV_DEG, img_w, img_h, DEM_RESOLUTION)

            # Resize photo to match DEM tile footprint (no arbitrary scale factor)
            search_img = cv2.resize(img_gray, (tile_w, tile_h))

            best_psr = -1
            best_col_off = 0
            best_row_off = 0

            for r in range(0, hs.shape[0] - tile_h, STEP_PX):
                for c in range(0, hs.shape[1] - tile_w, STEP_PX):
                    tile = hs[r:r + tile_h, c:c + tile_w]
                    dx, dy, psr = matcher.match(search_img, tile, edge_match=True)
                    if psr > best_psr:
                        best_psr  = psr
                        best_col_off = c
                        best_row_off = r

            # Convert best match back to world coords
            global_col = (cen_col - pad_px) + best_col_off + tile_w / 2
            global_row = (cen_row - pad_px) + best_row_off + tile_h / 2
            est_e, est_n = src.transform * (global_col, global_row)
            est_lon, est_lat = to_wgs84.transform(est_e, est_n)

            error_m = haversine_m(t_lat, t_lon, est_lat, est_lon)

            print(f"{photo_name}  GPS:({t_lat:.6f},{t_lon:.6f})  "
                  f"TRN:({est_lat:.6f},{est_lon:.6f})  "
                  f"err={error_m:.1f}m  PSR={best_psr:.2f}")

            rows.append({
                "photo": photo_name,
                "gps_lat": t_lat, "gps_lon": t_lon,
                "trn_lat": est_lat, "trn_lon": est_lon,
                "error_m": round(error_m, 1),
                "psr": round(best_psr, 2),
                "tile_footprint_m": f"{tile_w*DEM_RESOLUTION:.0f}x{tile_h*DEM_RESOLUTION:.0f}"
            })

    # ── Write CSV ─────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {OUT_CSV}")

    # ── Visual audit map ──────────────────────────────────────────────────
    hs_rgb = cv2.cvtColor(hs, cv2.COLOR_GRAY2RGB)
    im = Image.fromarray(hs_rgb).resize((1024, 1024), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    scale_x = 1024 / hs.shape[1]
    scale_y = 1024 / hs.shape[0]

    with rasterio.open(DEM_PATH) as src:
        to_mga = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        for row in rows:
            # GPS truth → green dot
            ge, gn = to_mga.transform(row["gps_lon"], row["gps_lat"])
            gc, gr = ~src.transform * (ge, gn)
            gx = (gc - (cen_col - pad_px)) * scale_x
            gy = (gr - (cen_row - pad_px)) * scale_y
            draw.ellipse([gx-5, gy-5, gx+5, gy+5], fill="lime", outline="lime")

            # TRN estimate → cyan dot
            te, tn = to_mga.transform(row["trn_lon"], row["trn_lat"])
            tc, tr = ~src.transform * (te, tn)
            tx = (tc - (cen_col - pad_px)) * scale_x
            ty = (tr - (cen_row - pad_px)) * scale_y
            draw.ellipse([tx-4, ty-4, tx+4, ty+4], fill="cyan", outline="cyan")

            # Error line
            draw.line([gx, gy, tx, ty], fill="red", width=1)

            # Label
            draw.text((gx+7, gy-8), f"{row['photo'][:8]} {row['error_m']}m", fill="white")

    # Legend
    draw.rectangle([5, 5, 200, 55], fill=(0, 0, 0, 180))
    draw.ellipse([12, 14, 22, 24], fill="lime")
    draw.text((26, 12), "GPS Truth", fill="lime")
    draw.ellipse([12, 34, 22, 44], fill="cyan")
    draw.text((26, 32), "TRN Estimate", fill="cyan")

    im.save(OUT_MAP)
    print(f"Map saved → {OUT_MAP}")

    errors = [r["error_m"] for r in rows]
    print(f"\n── Summary ──────────────────────────────")
    print(f"  Photos processed : {len(rows)}")
    print(f"  Mean error       : {sum(errors)/len(errors):.1f} m")
    print(f"  Min error        : {min(errors):.1f} m")
    print(f"  Max error        : {max(errors):.1f} m")
    print(f"  Tile footprint   : {rows[0]['tile_footprint_m']} m (at 150m AGL)")

if __name__ == "__main__":
    main()
