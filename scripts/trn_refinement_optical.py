"""
TRN Optical GPS-Aided Refinement Mode (Orthophoto)
==================================================
Identical logic to the DEM refinement, but specialized for 
high-resolution optical maps (e.g. 2m Orthophotos). 
Correlates optical textures (roads, roofs) rather than typography.
"""

import os, json, csv, math
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image, ImageDraw
from navigation.matcher import FFTMatcher

# ── Config ─────────────────────────────────────────────────────────────
PHOTO_DIR      = "Test photos 3"
ORTHO_PATH     = r"C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif"
GPS_TRUTH_FILE = "gps_ground_truth.json"
OUT_CSV        = "trn_optical_results.csv"
OUT_MAP        = "trn_optical_map.png"

AGL_M          = 150.0     # True AGL (m)
HFOV_DEG       = 69.7      # DJI Mini 3 True Horizontal FOV (Diagonal = 82.1)
ORTHO_RES      = 0.5       # Maps metres per pixel
MARGIN_M       = 100.0     # Context margin around footprint (metres) — sets max correction range
MAX_CORRECT_M  = 100.0     # Clamp correction to this (metres)
MIN_PSR        = 5.0       # Below this PSR, fall back to raw GPS

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def footprint_px(agl, hfov_deg, img_w, img_h, res):
    gnd_w = 2 * agl * math.tan(math.radians(hfov_deg / 2))
    gnd_h = gnd_w * (img_h / img_w)
    return max(1, int(gnd_w / res)), max(1, int(gnd_h / res))

def main():
    matcher = FFTMatcher()

    with open(GPS_TRUTH_FILE) as f:
        gps_truth = json.load(f)

    jpgs = sorted([k for k in gps_truth if k.endswith(".JPG")])
    print(f"TRN Optical Refinement | {len(jpgs)} photos | Map: {ORTHO_RES}m/px\n")

    rows = []

    with rasterio.open(ORTHO_PATH) as src:
        to_mga   = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        for photo_name in jpgs:
            gps = gps_truth[photo_name]
            gps_lat, gps_lon = gps["lat"], gps["lon"]

            # ── Load photo ──────────────────────────────────────────
            img_gray = cv2.imread(os.path.join(PHOTO_DIR, photo_name), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            img_h, img_w = img_gray.shape

            # ── Compute footprint ────────────────────────────────────
            tile_w, tile_h = footprint_px(AGL_M, HFOV_DEG, img_w, img_h, ORTHO_RES)

            # ── Extract Ortho tile + margin ─────────────────────────
            margin_px = int(MARGIN_M / ORTHO_RES)
            search_w = tile_w + 2 * margin_px
            search_h = tile_h + 2 * margin_px

            gps_e, gps_n = to_mga.transform(gps_lon, gps_lat)
            cen_col, cen_row = ~src.transform * (gps_e, gps_n)
            cen_col, cen_row = int(cen_col), int(cen_row)

            win = rasterio.windows.Window(
                cen_col - search_w // 2,
                cen_row - search_h // 2,
                search_w, search_h
            )
            
            ortho_patch_multiband = src.read(window=win)
            if ortho_patch_multiband.shape[1] == 0 or ortho_patch_multiband.shape[2] == 0:
                continue

            ortho_rgba = np.transpose(ortho_patch_multiband, (1, 2, 0))
            ortho_gray = cv2.cvtColor(ortho_rgba, cv2.COLOR_RGBA2GRAY)

            # ── Resize photo to its footprint size (anti-aliased) ────
            photo_resized = cv2.resize(img_gray, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

            # ── Slide photo over the Ortho tile ──────────────────────
            step = max(1, int(10.0 / ORTHO_RES)) # 10m steps dynamically scaled to map resolution
            best_psr = -1; best_dr = 0; best_dc = 0
            for dr in range(0, search_h - tile_h + 1, step):
                for dc in range(0, search_w - tile_w + 1, step):
                    patch = ortho_gray[dr:dr+tile_h, dc:dc+tile_w]
                    _, _, psr_c = matcher.match(photo_resized, patch, edge_match=True, native_size=True)
                    if psr_c > best_psr:
                        best_psr = psr_c; best_dr = dr; best_dc = dc
            psr = best_psr

            # Offset from centre of search window
            dc_centre = best_dc - margin_px
            dr_centre = best_dr - margin_px

            dx_m = dc_centre * ORTHO_RES   # positive = east
            dy_m = dr_centre * ORTHO_RES   # positive = south

            if psr < MIN_PSR:
                dx_m, dy_m = 0.0, 0.0

            corr_mag = math.sqrt(dx_m**2 + dy_m**2)
            if corr_mag > MAX_CORRECT_M:
                dx_m *= (MAX_CORRECT_M / corr_mag)
                dy_m *= (MAX_CORRECT_M / corr_mag)
                corr_mag = MAX_CORRECT_M

            ref_e = gps_e + dx_m
            ref_n = gps_n - dy_m   
            ref_lon, ref_lat = to_wgs84.transform(ref_e, ref_n)

            error_m = haversine_m(gps_lat, gps_lon, ref_lat, ref_lon)

            print(f"{photo_name}  err={corr_mag:5.1f}m  PSR={psr:5.2f}")

            rows.append({
                "photo":    photo_name,
                "gps_lat":  gps_lat, "gps_lon":  gps_lon,
                "trn_lat":  ref_lat, "trn_lon":  ref_lon,
                "correction_m": round(corr_mag, 1),
                "psr":      round(psr, 2),
                "fallback": psr < MIN_PSR,
            })

    # ── CSV ───────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ── Visual map (Drawn directly on the Orthophoto) ─────────────────
    all_lats = [r["gps_lat"] for r in rows]; all_lons = [r["gps_lon"] for r in rows]
    cen_lat = sum(all_lats) / len(all_lats); cen_lon = sum(all_lons) / len(all_lons)
    PAD_M   = 400

    with rasterio.open(ORTHO_PATH) as src:
        to_mga = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        cen_e, cen_n = to_mga.transform(cen_lon, cen_lat)
        pad_px = int(PAD_M / ORTHO_RES)
        cen_col, cen_row = ~src.transform * (cen_e, cen_n)
        cen_col, cen_row = int(cen_col), int(cen_row)

        win = rasterio.windows.Window(cen_col-pad_px, cen_row-pad_px, pad_px*2, pad_px*2)
        ortho_bg = src.read(window=win)
        
        # Convert RGBA band array to RGB image
        bg_rgb = cv2.cvtColor(np.transpose(ortho_bg, (1, 2, 0)), cv2.COLOR_RGBA2RGB)
        SIZE = 1200
        im   = Image.fromarray(bg_rgb).resize((SIZE, SIZE), Image.LANCZOS)
        draw = ImageDraw.Draw(im)
        sx = SIZE / (pad_px * 2); sy = SIZE / (pad_px * 2)

        def to_canvas(lat, lon):
            e, n = to_mga.transform(lon, lat)
            col, row = ~src.transform * (e, n)
            return (col - (cen_col-pad_px))*sx, (row - (cen_row-pad_px))*sy

        gps_pts = [to_canvas(r["gps_lat"], r["gps_lon"]) for r in rows]
        trn_pts = [to_canvas(r["trn_lat"], r["trn_lon"]) for r in rows]

        for i in range(len(gps_pts)-1):
            draw.line([gps_pts[i], gps_pts[i+1]], fill="lime", width=3)
        for i in range(len(trn_pts)-1):
            draw.line([trn_pts[i], trn_pts[i+1]], fill="#00FFFF", width=3)

        for i, r in enumerate(rows):
            gx, gy = gps_pts[i]; tx, ty = trn_pts[i]
            draw.line([gx, gy, tx, ty], fill="red", width=2)
            draw.ellipse([gx-6,gy-6,gx+6,gy+6], fill="lime")
            draw.ellipse([tx-5,ty-5,tx+5,ty+5], fill="yellow" if r["fallback"] else "#00FFFF")
            if i % 3 == 0:
                lbl = f"{r['photo'][:8]} {r['correction_m']:.0f}m"
                # Black outline for visibility on bright map
                draw.text((gx+7, gy-10), lbl, fill="black", stroke_width=2, stroke_fill="white")

        draw.rectangle([5,5,250,75], fill=(0,0,0, 200))
        draw.ellipse([12,14,22,24], fill="lime");   draw.text((26,12), "GPS position", fill="lime")
        draw.ellipse([12,34,22,44], fill="#00FFFF");   draw.text((26,32), "Optical TRN Refined", fill="#00FFFF")

        im.save(OUT_MAP)

    # ── Summary ───────────────────────────────────────────────────────
    corrections = [r["correction_m"] for r in rows]
    psrs        = [r["psr"] for r in rows]
    fallbacks   = sum(1 for r in rows if r["fallback"])

    print(f"\n{'='*55}")
    print(f"  OPTICAL TRN SUMMARY  ({len(rows)} photos)")
    print(f"{'='*55}")
    print(f"  Mean correction applied : {sum(corrections)/len(corrections):.1f} m")
    print(f"  Mean PSR                : {sum(psrs)/len(psrs):.2f}")
    print(f"  GPS fallbacks           : {fallbacks}/{len(rows)}")
    print(f"  Map Resolution          : {ORTHO_RES}m/px")
    print(f"  Map Saved               : {OUT_MAP}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
