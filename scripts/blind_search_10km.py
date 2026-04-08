import os
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from navigation.matcher import FFTMatcher

def generate_hillshade(dem_patch):
    dx, dy = np.gradient(dem_patch.astype(np.float32))
    dx, dy = np.clip(dx / 5.0, -10, 10), np.clip(dy / 5.0, -10, 10)
    res = 255 * (np.sin(0.7) * np.cos(np.arctan(np.sqrt(dx**2 + dy**2))) + 1) / 2
    return np.clip(res, 0, 255).astype(np.uint8)

def main():
    DEM_PATH, PHOTO_DIR = "5m_DEM.tif", "Test photos"
    matcher = FFTMatcher(target_size=(800, 800))
    photos = sorted([f for f in os.listdir(PHOTO_DIR) if f.lower().endswith('.jpg')])
    
    with rasterio.open(DEM_PATH) as src:
        # Full scan of the Red Rock headland (10km x 10km)
        # Search area: Easting 515000-525000, Northing 6680000-6690000
        left, bottom, right, top = 515000, 6680000, 525000, 6690000
        px_l, px_t = ~src.transform * (left, top)
        px_r, px_b = ~src.transform * (right, bottom)
        
        win = rasterio.windows.Window(int(px_l), int(px_t), int(px_r-px_l), int(px_b-px_t))
        full_dem = src.read(1, window=win)
        rows_s, cols_s = int(px_t), int(px_l)
        src_trans, src_crs = src.transform, src.crs
        to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    print("\n--- GLOBAL DEEP-SCAN (UNANCHORED) ---")
    
    for p_name in photos:
        print(f"Deep-Scanning {p_name}...")
        img = cv2.cvtColor(cv2.imread(os.path.join(PHOTO_DIR, p_name)), cv2.COLOR_BGR2GRAY)
        best_winner = None; best_psr = 0
        
        # Coarse sweep 50px steps
        for r_off in range(0, full_dem.shape[0] - 400, 20):
            for c_off in range(0, full_dem.shape[1] - 400, 20):
                tile_dem = full_dem[r_off:r_off+400, c_off:c_off+400]
                if np.mean(tile_dem < -100) > 0.2: continue # Ocean skip
                
                tile_hs = cv2.resize(generate_hillshade(tile_dem), (800, 800))
                
                # Match against the "True Signature" Scale
                dx, dy, psr = matcher.match(cv2.resize(img, (int(800*0.15), int(800*0.15*0.75))), tile_hs, edge_match=True)
                
                if psr > best_psr:
                    best_psr = psr
                    gr = rows_s + r_off + (400 + dy/2.0)
                    gc = cols_s + c_off + (400 + dx/2.0)
                    best_winner = (gc, gr)
        
        if best_winner:
            e, n = src_trans * (best_winner[0], best_winner[1])
            lo, la = to_wgs84.transform(e, n)
            print(f"-> DISCOVERY: {la:.6f}, {lo:.6f} | PSR: {best_psr:.2f}")

if __name__ == "__main__":
    main()
