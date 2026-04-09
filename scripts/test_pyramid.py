import os, json, cv2, math, re
import numpy as np
import rasterio
from pyproj import Transformer

PHOTO_DIR      = "Test photos 3"
ORTHO_PATH     = r"C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif"
GPS_TRUTH_FILE = "gps_ground_truth.json"

ORTHO_RES      = 0.5   
AGL_M          = 150.0 
HFOV_DEG       = 69.7  
MARGIN_M       = 100.0   

def footprint_px(agl_m, hfov_deg, img_w, img_h, ortho_res):
    fov_w_m = 2.0 * agl_m * math.tan(math.radians(hfov_deg / 2.0))
    fov_h_m = fov_w_m * (img_h / img_w)
    return int(fov_w_m / ortho_res), int(fov_h_m / ortho_res)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def main():
    with open(GPS_TRUTH_FILE, 'r') as f: truth_db = json.load(f)
    src = rasterio.open(ORTHO_PATH)
    to_mga = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:28356", "EPSG:4326", always_xy=True)

    for p in ['DJI_0068.JPG', 'DJI_0076.JPG', 'DJI_0083.JPG', 'DJI_0085.JPG']:
        gps = truth_db[p]
        gps_lat, gps_lon = gps["lat"], gps["lon"]

        img_gray = cv2.imread(os.path.join(PHOTO_DIR, p), cv2.IMREAD_GRAYSCALE)
        if img_gray is None: continue

        yaw_deg, agl_m = 0.0, AGL_M
        with open(os.path.join(PHOTO_DIR, p), 'rb') as fp:
            xstr = fp.read()[:50000].decode('utf-8', 'ignore')
            y = re.findall(r'FlightYawDegree=["\']?([+-]?[0-9.]+)', xstr, re.I)
            if y: yaw_deg = float(y[0])

        # Mathematical Rotation Fix (Verified)
        angle = -yaw_deg
        h, w = img_gray.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img_gray = cv2.warpAffine(img_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        img_h, img_w = img_gray.shape
        tile_w, tile_h = footprint_px(agl_m, HFOV_DEG, img_w, img_h, ORTHO_RES)

        # ── MAP THE FULL PATCH ──
        margin_px = int(MARGIN_M / ORTHO_RES)
        search_w = tile_w + 2 * margin_px
        search_h = tile_h + 2 * margin_px

        gps_e, gps_n = to_mga.transform(gps_lon, gps_lat)
        cen_col, cen_row = ~src.transform * (gps_e, gps_n)
        cen_col, cen_row = int(cen_col), int(cen_row)

        win = rasterio.windows.Window(cen_col - search_w//2, cen_row - search_h//2, search_w, search_h)
        patch = src.read(window=win)
        ortho_gray = cv2.cvtColor(np.transpose(patch, (1,2,0)), cv2.COLOR_RGBA2GRAY)

        photo_resized = cv2.resize(img_gray, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        # ── MACRO PASS (DOWN-SAMPLED 4x Pyramids) ──
        # Reduce size by exactly 4x (Resolves massive structural blobs like street curves, erasing bushes)
        macro_sc = 0.25
        macro_tile_w = int(tile_w * macro_sc)
        macro_tile_h = int(tile_h * macro_sc)
        macro_photo = cv2.resize(photo_resized, (macro_tile_w, macro_tile_h), interpolation=cv2.INTER_AREA)
        macro_ortho = cv2.resize(ortho_gray, (0,0), fx=macro_sc, fy=macro_sc, interpolation=cv2.INTER_AREA)

        # High-blur for Canny edge on Macro
        pb_m = cv2.GaussianBlur(macro_photo, (11, 11), 0)
        ob_m = cv2.GaussianBlur(macro_ortho, (11, 11), 0)
        pe_m = cv2.Canny(pb_m, 50, 150)
        oe_m = cv2.Canny(ob_m, 50, 150)
        
        # Thicken vectors massively
        k = np.ones((5,5), np.uint8)
        pe_m = cv2.dilate(pe_m, k, iterations=1)
        oe_m = cv2.dilate(oe_m, k, iterations=1)

        res_m = cv2.matchTemplate(oe_m, pe_m, cv2.TM_CCOEFF_NORMED)
        min_v, max_v_macro, min_l, max_l_macro = cv2.minMaxLoc(res_m)

        # Macro Coordinates (Top-Left of the photo overlay)
        macro_dc = max_l_macro[0]
        macro_dr = max_l_macro[1]

        # Upscale Macro prediction directly back to 100% pixel scale mapping
        upscaled_dc = int(macro_dc / macro_sc)
        upscaled_dr = int(macro_dr / macro_sc)

        dx_m = (upscaled_dc - margin_px) * ORTHO_RES
        dy_m = (upscaled_dr - margin_px) * ORTHO_RES
        
        ref_e, ref_n = gps_e + dx_m, gps_n - dy_m   
        ref_lon, ref_lat = to_wgs84.transform(ref_e, ref_n)
        error_m = haversine_m(gps_lat, gps_lon, ref_lat, ref_lon)

        print(f"{p}: Macro-Score={max_v_macro:.3f} | Error={error_m:5.1f}m")

if __name__ == '__main__': main()
