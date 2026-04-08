import os
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from navigation.matcher import FFTMatcher

def generate_hillshade(dem_patch, azimuth=340, elevation=45):
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)
    dx, dy = np.gradient(dem_patch)
    dx = np.clip(dx / 5.0, -100, 100)
    dy = np.clip(dy / 5.0, -100, 100)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(dx, -dy)
    shaded = (np.sin(el_rad) * np.cos(slope)) + \
             (np.cos(el_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    res = 255 * (shaded + 1) / 2
    return np.clip(res, 0, 255).astype(np.uint8)

def main():
    DEM_PATH = "5m_DEM.tif"
    PHOTO_DIR = "Test photos"
    
    # Target: Red Rock River Mouth
    CENTER_E = 521993
    CENTER_N = 6682633
    SEARCH_RADIUS = 1000 
    
    matcher = FFTMatcher(target_size=(800, 800))
    
    print(f"Loading DEM...")
    with rasterio.open(DEM_PATH) as src:
        row_start, col_start = ~src.transform * (CENTER_E - SEARCH_RADIUS, CENTER_N + SEARCH_RADIUS)
        row_end, col_end = ~src.transform * (CENTER_E + SEARCH_RADIUS, CENTER_N - SEARCH_RADIUS)
        window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
        dem_data = src.read(1, window=window)
        src_crs = src.crs
        
    map_tile_raw = generate_hillshade(dem_data)
    map_tile = cv2.resize(map_tile_raw, (800, 800))
    
    photos = [f for f in os.listdir(PHOTO_DIR) if f.lower().endswith('.jpg')]
    p_name = photos[5]
    photo_path = os.path.join(PHOTO_DIR, p_name)
    dji_img = cv2.imread(photo_path)
    gray = cv2.cvtColor(dji_img, cv2.COLOR_BGR2GRAY)
    
    best_psr, best_padded, best_dx, best_dy, best_scale = 0, None, 0, 0, 0
    
    print("Finding best match scale...")
    for scale in [0.2, 0.3, 0.4, 0.5]:
        h, w = gray.shape
        w_s = int(800 * scale)
        h_s = int(w_s * (h/w))
        resized = cv2.resize(gray, (w_s, h_s))
        padded = np.zeros((800, 800), dtype=np.uint8)
        y_o, x_o = (800 - h_s) // 2, (800 - w_s) // 2
        padded[y_o:y_o+h_s, x_o:x_o+w_s] = resized
        
        dx, dy, psr = matcher.match(padded, map_tile, denoise=True)
        if psr > best_psr:
            best_psr, best_padded, best_dx, best_dy, best_scale = psr, padded, dx, dy, scale

    # MGA to GPS
    match_e = CENTER_E + (best_dx * 2.5)
    match_n = CENTER_N - (best_dy * 2.5)
    trans = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    match_lon, match_lat = trans.transform(match_e, match_n)

    # Canvas: 1720 x 900
    cv = np.zeros((900, 1720, 3), dtype=np.uint8)
    
    left_img = cv2.cvtColor(cv2.GaussianBlur(best_padded, (15, 15), 0), cv2.COLOR_GRAY2BGR)
    right_img = cv2.cvtColor(map_tile, cv2.COLOR_GRAY2BGR)
    
    m_x, m_y = 400 + best_dx, 400 + best_dy
    box_s = int(800 * best_scale)
    cv2.rectangle(right_img, (int(m_x - box_s//2), int(m_y - box_s//2)), 
                 (int(m_x + box_s//2), int(m_y + box_s//2)), (0, 255, 0), 3)
    cv2.drawMarker(right_img, (int(m_x), int(m_y)), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

    cv[50:850, 50:850] = left_img
    cv[50:850, 870:1670] = right_img

    # Original Photo Inset (TOP RIGHT of Left Panel)
    inset_w = 260
    inset = cv2.resize(dji_img, (inset_w, int(inset_w * 0.5625)))
    # Place at Top-Right of Left Panel (Panel ends at 850)
    cv[60:60+inset.shape[0], 580:580+inset.shape[1]] = inset
    cv2.rectangle(cv, (580, 60), (580+inset.shape[1], 60+inset.shape[0]), (255, 255, 255), 2)
    cv2.putText(cv, "RAW SOURCE", (590, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(cv, "DRONE FEED (DENOISED)", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(cv, "TERRAIN MATCH (DEM)", (870, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Readout Box
    rect_w = 420
    cv2.rectangle(cv, (650, 750), (650+rect_w, 890), (20, 20, 20), -1)
    cv2.rectangle(cv, (650, 750), (650+rect_w, 890), (0, 255, 0), 2)
    
    y_p = 780
    cv2.putText(cv, "MATCH READY: LOCK ACQUIRED", (660, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA); y_p += 30
    cv2.putText(cv, f"CONFIDENCE: {best_psr:.2f} PSR", (660, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); y_p += 25
    cv2.putText(cv, f"MATCH LAT: {match_lat:.6f}", (660, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); y_p += 25
    cv2.putText(cv, f"MATCH LON: {match_lon:.6f}", (660, y_p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # GPS Tags on images
    cv2.putText(cv, "GPS: [NO_SIGNAL]", (60, 835), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(cv, f"GPS: {match_lat:.4f} / {match_lon:.4f} (AUTO)", (880, 835), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    out_name = "tactical_match_final.png"
    cv2.imwrite(out_name, cv)
    print(f"Final Cinematic Comparison saved to {out_name}")

if __name__ == "__main__":
    main()
