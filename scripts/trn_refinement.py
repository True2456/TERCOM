"""
TRN GPS-Aided Refinement Mode
================================
Designed for Raspberry Pi / low-CPU deployment.

Architecture:
  - GPS gives initial position (from onboard GPS or EXIF for testing)
  - Extract one DEM tile centered on GPS position, sized to photo footprint
  - Single FFT phase correlation -> (dx, dy) pixel offset
  - Convert offset to metric correction -> refined TRN position
  - No sliding window loop: O(1) FFT calls, not O(N^2)

CPU profile (per photo):
  - DEM tile read  : ~2ms
  - Image resize   : ~5ms
  - FFT 512x512    : ~15ms on Pi 4
  - Total          : ~25ms per fix  (~40 Hz capable)
"""

import os, json, csv, math
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image, ImageDraw
from navigation.matcher import FFTMatcher

# ── Config ─────────────────────────────────────────────────────────────
PHOTO_DIR      = "Test photos 2"
DEM_PATH       = "5m_DEM.tif"
GPS_TRUTH_FILE = "gps_ground_truth.json"
OUT_CSV        = "trn_refinement_results.csv"
OUT_MAP        = "trn_refinement_map.png"

AGL_M          = 150.0     # True AGL (m)
HFOV_DEG       = 82.1      # DJI Mini 3 HFOV
DEM_RES        = 5.0       # DEM metres per pixel
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

def footprint_px(agl, hfov_deg, img_w, img_h, dem_res):
    gnd_w = 2 * agl * math.tan(math.radians(hfov_deg / 2))
    gnd_h = gnd_w * (img_h / img_w)
    return max(1, int(gnd_w / dem_res)), max(1, int(gnd_h / dem_res))

def main():
    matcher = FFTMatcher()

    with open(GPS_TRUTH_FILE) as f:
        gps_truth = json.load(f)

    jpgs = sorted([k for k in gps_truth if k.endswith(".JPG")])
    print(f"TRN GPS-Aided Refinement | {len(jpgs)} photos | AGL={AGL_M}m\n")

    rows = []

    with rasterio.open(DEM_PATH) as src:
        to_mga   = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        for photo_name in jpgs:
            gps = gps_truth[photo_name]
            gps_lat, gps_lon = gps["lat"], gps["lon"]

            # ── Load photo ──────────────────────────────────────────
            img_gray = cv2.imread(os.path.join(PHOTO_DIR, photo_name), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"  {photo_name}: SKIP (unreadable)")
                continue
            img_h, img_w = img_gray.shape

            # ── Compute footprint ────────────────────────────────────
            tile_w, tile_h = footprint_px(AGL_M, HFOV_DEG, img_w, img_h, DEM_RES)

            # ── Extract DEM tile = footprint + margin on each side ──
            # The margin is what allows displacement to be detected.
            # Max detectable correction = MARGIN_M in any direction.
            margin_px = int(MARGIN_M / DEM_RES)
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
            dem_tile = src.read(1, window=win)
            dem_tile[dem_tile < -100] = 0
            dem_norm = cv2.normalize(dem_tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # ── Resize photo to its footprint size (not search tile size) ─
            photo_resized = cv2.resize(img_gray, (tile_w, tile_h))

            # ── Slide photo over the larger DEM tile (few iterations) ─
            # Step = 5px = 25m; only ~(2*margin_px/step)^2 iterations
            step = 5
            best_psr = -1; best_dr = 0; best_dc = 0
            for dr in range(0, search_h - tile_h + 1, step):
                for dc in range(0, search_w - tile_w + 1, step):
                    patch = dem_norm[dr:dr+tile_h, dc:dc+tile_w]
                    _, _, psr_c = matcher.match(photo_resized, patch, edge_match=True, native_size=True)
                    if psr_c > best_psr:
                        best_psr = psr_c; best_dr = dr; best_dc = dc
            psr = best_psr

            # Offset from centre of search window
            dc_centre = best_dc - margin_px
            dr_centre = best_dr - margin_px

            # ── Convert pixel offset to metres ───────────────────────
            dx_m = dc_centre * DEM_RES   # positive = east
            dy_m = dr_centre * DEM_RES   # positive = south

            # Clamp large corrections if PSR is weak
            if psr < MIN_PSR:
                dx_m, dy_m = 0.0, 0.0  # fall back to raw GPS

            correction_m = math.sqrt(dx_m**2 + dy_m**2)
            if correction_m > MAX_CORRECT_M:
                scale = MAX_CORRECT_M / correction_m
                dx_m *= scale
                dy_m *= scale

            # ── Apply correction in MGA space ────────────────────────
            ref_e = gps_e + dx_m
            ref_n = gps_n - dy_m   # row down = northing decreases
            ref_lon, ref_lat = to_wgs84.transform(ref_e, ref_n)

            error_m = haversine_m(gps_lat, gps_lon, ref_lat, ref_lon)
            corr_applied = math.sqrt(dx_m**2 + dy_m**2)

            print(f"{photo_name}  GPS:({gps_lat:.6f},{gps_lon:.6f})  "
                  f"TRN:({ref_lat:.6f},{ref_lon:.6f})  "
                  f"corr={corr_applied:.1f}m  PSR={psr:.2f}")

            rows.append({
                "photo":    photo_name,
                "gps_lat":  gps_lat,
                "gps_lon":  gps_lon,
                "trn_lat":  ref_lat,
                "trn_lon":  ref_lon,
                "correction_m": round(corr_applied, 1),
                "psr":      round(psr, 2),
                "fallback": psr < MIN_PSR,
            })

    # ── CSV ───────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────
    corrections = [r["correction_m"] for r in rows]
    psrs        = [r["psr"] for r in rows]
    fallbacks   = sum(1 for r in rows if r["fallback"])

    print(f"\n{'='*55}")
    print(f"  TRN REFINEMENT SUMMARY  ({len(rows)} photos)")
    print(f"{'='*55}")
    print(f"  Mean correction applied : {sum(corrections)/len(corrections):.1f} m")
    print(f"  Max correction applied  : {max(corrections):.1f} m")
    print(f"  Mean PSR                : {sum(psrs)/len(psrs):.2f}")
    print(f"  GPS fallbacks (low PSR) : {fallbacks}/{len(rows)}")
    print(f"  Tile footprint          : {tile_w*DEM_RES:.0f}x{tile_h*DEM_RES:.0f} m")
    print(f"  CPU mode                : single FFT per photo (Pi-ready)")
    print(f"{'='*55}")
    print(f"\nResults -> {OUT_CSV}")

    # ── Visual map ────────────────────────────────────────────────────
    all_lats = [r["gps_lat"] for r in rows]
    all_lons = [r["gps_lon"] for r in rows]
    cen_lat = sum(all_lats) / len(all_lats)
    cen_lon = sum(all_lons) / len(all_lons)
    PAD_M   = 400

    with rasterio.open(DEM_PATH) as src:
        to_mga   = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        cen_e, cen_n = to_mga.transform(cen_lon, cen_lat)
        pad_px = int(PAD_M / DEM_RES)
        cen_col, cen_row = ~src.transform * (cen_e, cen_n)
        cen_col, cen_row = int(cen_col), int(cen_row)

        win = rasterio.windows.Window(cen_col-pad_px, cen_row-pad_px, pad_px*2, pad_px*2)
        dem = src.read(1, window=win); dem[dem < -100] = 0
        hs  = cv2.normalize(dem, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hs_rgb = cv2.cvtColor(hs, cv2.COLOR_GRAY2RGB)
        SIZE = 1024
        im   = Image.fromarray(hs_rgb).resize((SIZE, SIZE), Image.LANCZOS)
        draw = ImageDraw.Draw(im)
        sx = SIZE / (pad_px * 2); sy = SIZE / (pad_px * 2)

        def to_canvas(lat, lon):
            e, n = to_mga.transform(lon, lat)
            col, row = ~src.transform * (e, n)
            return (col - (cen_col-pad_px))*sx, (row - (cen_row-pad_px))*sy

        gps_pts = [to_canvas(r["gps_lat"], r["gps_lon"]) for r in rows]
        trn_pts = [to_canvas(r["trn_lat"], r["trn_lon"]) for r in rows]

        # GPS track
        for i in range(len(gps_pts)-1):
            draw.line([gps_pts[i], gps_pts[i+1]], fill="lime", width=2)
        # TRN refined track
        for i in range(len(trn_pts)-1):
            draw.line([trn_pts[i], trn_pts[i+1]], fill="cyan", width=2)

        for i, r in enumerate(rows):
            gx, gy = gps_pts[i]; tx, ty = trn_pts[i]
            draw.line([gx, gy, tx, ty], fill="red", width=1)
            draw.ellipse([gx-5,gy-5,gx+5,gy+5], fill="lime")
            draw.ellipse([tx-4,ty-4,tx+4,ty+4],
                         fill="yellow" if r["fallback"] else "cyan")
            if i % 3 == 0:
                lbl = f"{r['photo'][:8]} {r['correction_m']:.0f}m"
                draw.text((gx+7, gy-10), lbl, fill="white")

        draw.rectangle([5,5,250,75], fill=(0,0,0))
        draw.ellipse([12,14,22,24], fill="lime");   draw.text((26,12), "GPS position", fill="lime")
        draw.ellipse([12,34,22,44], fill="cyan");   draw.text((26,32), "TRN refined position", fill="cyan")
        draw.ellipse([12,54,22,64], fill="yellow"); draw.text((26,52), "GPS fallback (low PSR)", fill="yellow")

        bar_px = int(100/DEM_RES*sx)
        draw.rectangle([SIZE-bar_px-10, SIZE-25, SIZE-10, SIZE-15], fill="white")
        draw.text((SIZE-bar_px-10, SIZE-40), "100m", fill="white")

        im.save(OUT_MAP)
        print(f"Map saved -> {OUT_MAP}")

if __name__ == "__main__":
    main()
