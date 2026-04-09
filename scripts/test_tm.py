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

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def footprint_px(agl_m, hfov_deg, img_w, img_h, ortho_res):
    fov_w_m = 2.0 * agl_m * math.tan(math.radians(hfov_deg / 2.0))
    fov_h_m = fov_w_m * (img_h / img_w)
    return int(fov_w_m / ortho_res), int(fov_h_m / ortho_res)

def main():
    with open(GPS_TRUTH_FILE, 'r') as f:
        truth_db = json.load(f)
    src = rasterio.open(ORTHO_PATH)
    to_mga = Transformer.from_crs("EPSG:4326", "EPSG:28356", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:28356", "EPSG:4326", always_xy=True)

    results = []

    for photo_name in sorted(os.listdir(PHOTO_DIR)):
        if not photo_name.endswith('.JPG'): continue
        if photo_name not in truth_db: continue
        gps = truth_db[photo_name]
        gps_lat, gps_lon = gps["lat"], gps["lon"]

        img_gray = cv2.imread(os.path.join(PHOTO_DIR, photo_name), cv2.IMREAD_GRAYSCALE)
        if img_gray is None: continue

        yaw_deg, agl_m = 0.0, AGL_M
        with open(os.path.join(PHOTO_DIR, photo_name), 'rb') as fp:
            xstr = fp.read()[:50000].decode('utf-8', 'ignore')
            y = re.findall(r'FlightYawDegree=["\']?([+-]?[0-9.]+)', xstr, re.I)
            if y: yaw_deg = float(y[0])
            a = re.findall(r'RelativeAltitude=["\']?([+-]?[0-9.]+)', xstr, re.I)
            if a: agl_m = abs(float(a[0]))
            else:
                a2 = re.findall(r'Altitude=["\']?([+-]?[0-9.]+)', xstr, re.I)
                if a2: agl_m = abs(float(a2[-1]))

        if abs(yaw_deg) > 0.1:
            h, w = img_gray.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -yaw_deg, 1.0)
            img_gray = cv2.warpAffine(img_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        img_h, img_w = img_gray.shape
        tile_w, tile_h = footprint_px(agl_m, HFOV_DEG, img_w, img_h, ORTHO_RES)

        margin_px = int(MARGIN_M / ORTHO_RES)
        search_w = tile_w + 2 * margin_px
        search_h = tile_h + 2 * margin_px

        gps_e, gps_n = to_mga.transform(gps_lon, gps_lat)
        cen_col, cen_row = ~src.transform * (gps_e, gps_n)
        cen_col, cen_row = int(cen_col), int(cen_row)

        win = rasterio.windows.Window(cen_col - search_w // 2, cen_row - search_h // 2, search_w, search_h)
        ortho_patch = src.read(window=win)
        if ortho_patch.shape[1] == 0 or ortho_patch.shape[2] == 0: continue
        ortho_gray = cv2.cvtColor(np.transpose(ortho_patch, (1,2,0)), cv2.COLOR_RGBA2GRAY)

        photo_resized = cv2.resize(img_gray, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        # Apply Canny edge detection for structural matching instead of raw pixel density
        p_blur = cv2.GaussianBlur(photo_resized, (5, 5), 0)
        o_blur = cv2.GaussianBlur(ortho_gray, (5, 5), 0)
        p_edge = cv2.Canny(p_blur, 50, 150)
        o_edge = cv2.Canny(o_blur, 50, 150)
        # Thicken
        kernel = np.ones((3,3), np.uint8)
        p_edge = cv2.dilate(p_edge, kernel, iterations=1)
        o_edge = cv2.dilate(o_edge, kernel, iterations=1)

        # Template Match CCOEFF_NORMED
        res = cv2.matchTemplate(o_edge, p_edge, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # max_loc is (x, y) = (col, row) of the top-left corner of the matched patch
        dc = max_loc[0]
        dr = max_loc[1]

        dc_centre = dc - margin_px
        dr_centre = dr - margin_px

        dx_m = dc_centre * ORTHO_RES
        dy_m = dr_centre * ORTHO_RES

        corr_mag = math.sqrt(dx_m**2 + dy_m**2)

        ref_e = gps_e + dx_m
        ref_n = gps_n - dy_m   
        ref_lon, ref_lat = to_wgs84.transform(ref_e, ref_n)
        error_m = haversine_m(gps_lat, gps_lon, ref_lat, ref_lon)

        print(f"{photo_name}: error={corr_mag:5.1f}m score={max_val:.3f}")
        results.append((gps_lat, gps_lon, ref_lat, ref_lon, photo_name, gps_e, gps_n, ref_e, ref_n))

    if len(results) == 0: return

    print("Generating visualization map...")
    # Calculate bounding box in MGA coordinates
    l_e = min([r[5] for r in results] + [r[7] for r in results])
    r_e = max([r[5] for r in results] + [r[7] for r in results])
    t_n = max([r[6] for r in results] + [r[8] for r in results])
    b_n = min([r[6] for r in results] + [r[8] for r in results])

    pad = 200 # meters margin
    l_e, r_e = l_e - pad, r_e + pad
    b_n, t_n = b_n - pad, t_n + pad

    col_min, row_max = ~src.transform * (l_e, b_n)
    col_max, row_min = ~src.transform * (r_e, t_n)
    
    col_min, col_max = int(col_min), int(col_max)
    row_min, row_max = int(row_min), int(row_max)
    
    w = col_max - col_min
    h = row_max - row_min
    
    win = rasterio.windows.Window(col_min, row_min, w, h)
    ortho_img = src.read(window=win)
    ortho_img = np.transpose(ortho_img, (1,2,0)) # RGBA
    if ortho_img.shape[2] == 4:
        viz_map = cv2.cvtColor(ortho_img, cv2.COLOR_RGBA2BGR)
    else:
        viz_map = cv2.cvtColor(ortho_img, cv2.COLOR_RGB2BGR)

    def to_px(e, n):
        col, row = ~src.transform * (e, n)
        return int(col - col_min), int(row - row_min)

    prev_gps = None
    prev_ref = None
    
    for r in results:
        g_e, g_n, r_e, r_n, pname = r[5], r[6], r[7], r[8], r[4]
        gps_px = to_px(g_e, g_n)
        ref_px = to_px(r_e, r_n)
        
        # Red line for error
        cv2.line(viz_map, gps_px, ref_px, (0, 0, 255), 2)
        
        # Connect path dots
        if prev_gps:
            cv2.line(viz_map, prev_gps, gps_px, (0, 255, 0), 2)
            cv2.line(viz_map, prev_ref, ref_px, (255, 255, 0), 2)
        
        # Data points
        cv2.circle(viz_map, gps_px, 6, (0, 255, 0), -1)
        cv2.circle(viz_map, ref_px, 6, (255, 255, 0), -1)
        
        # Text label (draw text shadow for clarity)
        txt = f"{pname} {haversine_m(r[0], r[1], r[2], r[3]):.1f}m"
        cv2.putText(viz_map, txt, (gps_px[0]+10, gps_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(viz_map, txt, (gps_px[0]+10, gps_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        prev_gps = gps_px
        prev_ref = ref_px

    # Legend
    cv2.rectangle(viz_map, (10, 10), (300, 70), (0, 0, 0), -1)
    cv2.circle(viz_map, (30, 30), 6, (0, 255, 0), -1)
    cv2.putText(viz_map, "GPS Ground Truth", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(viz_map, (30, 55), 6, (255, 255, 0), -1)
    cv2.putText(viz_map, "Optical TRN Refined", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

    outpath = "trn_test_viz_map.png"
    cv2.imwrite(outpath, viz_map)
    print(f"Saved visualization to {outpath}!")

if __name__ == '__main__': main()
