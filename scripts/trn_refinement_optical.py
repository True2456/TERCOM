"""
TRN Optical GPS-Aided Refinement Mode (Orthophoto)
==================================================
Identical logic to the DEM refinement, but specialized for 
high-resolution optical maps (e.g. 2m Orthophotos). 
Correlates optical textures (roads, roofs) rather than typography.
"""

import os, json, csv, math, re
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image, ImageDraw
from navigation.matcher import FFTMatcher, ORBMatcher

# ── Config ─────────────────────────────────────────────────────────────
USE_ORB        = False
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
    if USE_ORB:
        matcher = ORBMatcher(max_features=1000)
    else:
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

            # ── Yaw & Altitude Extraction (Multi-Strategy Robust) ───────────
            yaw_deg = 0.0
            agl_m   = AGL_M
            with open(os.path.join(PHOTO_DIR, photo_name), 'rb') as fp:
                data = fp.read()
                s = data.find(b'<x:xmpmeta')
                e = data.find(b'</x:xmpmeta>')
                if s != -1 and e != -1:
                    xstr = data[s:e+12].decode('utf-8', 'ignore')
                    
                    # 1. Extraction: Yaw
                    yaw_matches = re.findall(r'FlightYawDegree=["\']?([+-]?[0-9.]+)', xstr, re.I)
                    if yaw_matches:
                        try: yaw_deg = float(yaw_matches[0])
                        except: pass

                    # 2. Extraction: Altitude (Relative -> Last Altitude -> Single Altitude)
                    alt_matches = re.findall(r'RelativeAltitude=["\']?([+-]?[0-9.]+)', xstr, re.I)
                    if alt_matches:
                        try: agl_m = abs(float(alt_matches[0]))
                        except: pass
                    else:
                        # Fallback: Many DJI photos log [Absolute, Relative] in generic 'Altitude' tags
                        gen_alts = re.findall(r'Altitude=["\']?([+-]?[0-9.]+)', xstr, re.I)
                        if len(gen_alts) >= 2:
                            try: agl_m = abs(float(gen_alts[-1]))
                            except: pass
                        elif gen_alts:
                            try: agl_m = abs(float(gen_alts[0]))
                            except: pass

            if abs(yaw_deg) > 0.1:
                # Rotate drone frame to North-Up
                angle = -yaw_deg
                (h, w) = img_gray.shape[:2]
                (cX, cY) = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                img_gray = cv2.warpAffine(img_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Read dims AFTER rotation — footprint must reflect rotated shape
            img_h, img_w = img_gray.shape

            # ── Compute footprint using live per-photo AGL ────────────
            tile_w, tile_h = footprint_px(agl_m, HFOV_DEG, img_w, img_h, ORTHO_RES)

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

            # ── Tracking Sequence ────────────────────────────────────
            if USE_ORB:
                # Direct Homography computation natively scans the absolute patch
                dr_offset, dc_offset, inliers = matcher.match(photo_resized, ortho_gray)
                best_dr = dr_offset
                best_dc = dc_offset
                psr = inliers
            else:
                # ── Normalized Cross-Correlation Template Matching ──────────
                # Apply Canny edge detection to isolate geometric structures
                p_blur = cv2.GaussianBlur(photo_resized, (5, 5), 0)
                o_blur = cv2.GaussianBlur(ortho_gray, (5, 5), 0)
                p_edge = cv2.Canny(p_blur, 50, 150)
                o_edge = cv2.Canny(o_blur, 50, 150)

                # Thicken structural vectors
                kernel = np.ones((3,3), np.uint8)
                p_edge = cv2.dilate(p_edge, kernel, iterations=1)
                o_edge = cv2.dilate(o_edge, kernel, iterations=1)

                # Template Match natively slides the photo across the ortho window
                res = cv2.matchTemplate(o_edge, p_edge, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                best_dc = max_loc[0]
                best_dr = max_loc[1]
                psr = max_val

            # Offset from centre of search window
            if USE_ORB:
                dc_centre = best_dc - (search_w / 2.0)
                dr_centre = best_dr - (search_h / 2.0)
            else:
                # Template match outputs the top-left coordinate! 
                # To align with the map center, subtract the margin.
                dc_centre = best_dc - margin_px
                dr_centre = best_dr - margin_px

            dx_m = dc_centre * ORTHO_RES   # positive = east
            dy_m = dr_centre * ORTHO_RES   # positive = south

            # Fallback if Match Confidence is too low
            is_fallback = False
            if USE_ORB:
                if psr < MIN_PSR:
                    is_fallback = True
            else:
                if psr < 0.03: # NORMED Template Match returns ~0.05 to ~0.50
                    is_fallback = True
            
            if is_fallback:
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

            print(f"{photo_name}  err={corr_mag:5.1f}m  PSR={psr:5.2f}  AGL={agl_m:.1f}m  Yaw={yaw_deg:.1f}°")

            rows.append({
                "photo":    photo_name,
                "gps_lat":  gps_lat, "gps_lon":  gps_lon,
                "trn_lat":  ref_lat, "trn_lon":  ref_lon,
                "correction_m": round(corr_mag, 1),
                "psr":      round(psr, 2),
                "fallback": is_fallback,
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
