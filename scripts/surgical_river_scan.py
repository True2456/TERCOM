import os
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from pyproj import CRS
from navigation.matcher import FFTMatcher
from PIL import Image, ImageDraw

def main():
    # SETTINGS
    DEM_PATH = "5m_DEM.tif"
    PHOTO_DIR = "Test photos"
    OUT_DIR = "River_Detection_Results"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # RIVER ANCHOR (Red Rock Park Street Public Toilet)
    # Using the exact Truth location you provided.
    ANCHOR_LAT, ANCHOR_LON = -29.98332853, 153.22689708
    SEARCH_RADIUS_M = 1200 # 2.4km search box to stay strictly in the river/town zone
    
    matcher = FFTMatcher()
    
    with rasterio.open(DEM_PATH) as src:
        to_mga = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        e, n = to_mga.transform(ANCHOR_LON, ANCHOR_LAT)
        
        # Crop the RIVER CORRIDOR Window
        win_size = int(SEARCH_RADIUS_M * 2 / 5) # pixels (5m resolution)
        px_c, px_r = ~src.transform * (e, n)
        window = rasterio.windows.Window(int(px_c - win_size//2), int(px_r - win_size//2), win_size, win_size)
        dem = src.read(1, window=window)
        
        # Clean NoData
        dem[dem < -100] = 0
        
        # INCREASE CONTRAST for flat terrain
        # We use a narrower elevation range to bring out the riverbanks
        dem_min, dem_max = np.min(dem), np.max(dem)
        hs = cv2.normalize(dem, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Process Photos
        photos = [f for f in os.listdir(PHOTO_DIR) if f.endswith(".jpg")]
        photos.sort()
        
        print(f"--- SURGICAL RIVER SCAN (RADIUS: {SEARCH_RADIUS_M}m) ---")
        
        output_results = []
        
        for photo_name in photos:
            photo_path = os.path.join(PHOTO_DIR, photo_name)
            img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            
            best_psr = 0
            best_coord = (0, 0)
            
            # SCALE LOCK: 0.15 (Validated for ~150m AGL)
            target_scale = 0.15
            h, w = img.shape
            sh, sw = int(h * target_scale), int(w * target_scale)
            search_img = cv2.resize(img, (sw, sh))
            
            # Scan with 20px step
            for r in range(0, hs.shape[0] - sh, 20):
                for c in range(0, hs.shape[1] - sw, 20):
                    tile = hs[r:r+sh, c:c+sw]
                    
                    # FIXED: Capture all 3 return values (dx, dy, psr)
                    dx, dy, psr = matcher.match(search_img, tile, edge_match=True)
                    
                    if psr > best_psr:
                        best_psr = psr
                        # Refine position with the FFT displacement (dx, dy)
                        best_coord = (r + dy, c + dx)
            
            # Convert to Global MGA
            gr, gc = best_coord[0] + window.row_off, best_coord[1] + window.col_off
            ge, gn = src.transform * (gc + sw/2, gr + sh/2)
            
            to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = to_wgs84.transform(ge, gn)
            
            print(f"{photo_name[:15]}... -> {lat:.6f}, {lon:.6f} | PSR: {best_psr:.2f}")
            output_results.append((lat, lon, best_psr))

        # FINAL VERIFICATION: Visual Track Mapping
        print("\nCreating River Alignment Map...")
        hs_rgb = cv2.cvtColor(hs, cv2.COLOR_GRAY2RGB)
        im = Image.fromarray(hs_rgb)
        draw = ImageDraw.Draw(im)
        
        # Plot TRUTH (Red X)
        rx, ry = px_c - window.col_off, px_r - window.row_off
        draw.line([rx-20, ry-20, rx+20, ry+20], fill="red", width=3)
        draw.line([rx+20, ry-20, rx-20, ry+20], fill="red", width=3)
        
        # Plot LOCATIONS (Cyan dots)
        for lat, lon, psr in output_results:
            ee, nn = to_mga.transform(lon, lat)
            pc, pr = ~src.transform * (ee, nn)
            lx, ly = pc - window.col_off, pr - window.row_off
            draw.ellipse([lx-5, ly-5, lx+5, ly+5], fill="cyan")
            
        im.save("mission_audit_v4_RIVER.png")
        print("River Audit Saved: mission_audit_v4_RIVER.png")

if __name__ == "__main__":
    main()
