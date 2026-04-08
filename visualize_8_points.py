import os
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image, ImageDraw

def generate_hillshade(dem_patch):
    dx, dy = np.gradient(dem_patch.astype(np.float32))
    dx, dy = np.clip(dx / 5.0, -10, 10), np.clip(dy / 5.0, -10, 10)
    res = 255 * (np.sin(0.7) * np.cos(np.arctan(np.sqrt(dx**2 + dy**2))) + 1) / 2
    return np.clip(res, 0, 255).astype(np.uint8)

def main():
    DEM_PATH = "5m_DEM.tif"
    OUT_PATH = "mission_audit_v3_DISCOVERY.png"
    
    # NEW NATURAL DISCOVERIES (Unanchored)
    discoveries = [
        (-29.975651, 153.223489), # 1 (Rock)
        (-29.982700, 153.217700), # 2 (Rock)
        (-29.981524, 153.244727)  # 3 (False Positive?) 
    ]
    
    # USER TRUTH (The "Proposed" Toilet)
    USER_LAT, USER_LON = -29.9833285309228, 153.22689708868106
    
    with rasterio.open(DEM_PATH) as src:
        to_mga = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        # Center on the ROCK area (West of the previous center)
        # Shift 1km West/North
        cent_e, cent_n = to_mga.transform(USER_LON - 0.01, USER_LAT + 0.01)
        px_c, px_r = ~src.transform * (cent_e, cent_n)
        
        win_size = 3000 # 15km / 5m
        win = rasterio.windows.Window(int(px_c - win_size//2), int(px_r - win_size//2), win_size, win_size)
        dem = src.read(1, window=win)
        clean_dem = dem.copy(); clean_dem[clean_dem < -100] = 0
        hs = generate_hillshade(clean_dem)
        
        im = Image.fromarray(hs).convert("RGB")
        draw = ImageDraw.Draw(im)
        
        # 1. Plot USER TRUTH (Red X)
        te, tn = to_mga.transform(USER_LON, USER_LAT)
        ty, tx = ~src.transform * (te, tn)
        rx, ry = ty - (px_c - win_size//2), tx - (px_r - win_size//2)
        draw.line([rx-20, ry-20, rx+20, ry+20], fill="red", width=5)
        draw.line([rx+20, ry-20, rx-20, ry+20], fill="red", width=5)
        draw.text((rx+25, ry), "USER TRUTH (Toilet?)", fill="red")
        
        # 2. Plot DISCOVERIES (Green Circles for rocks)
        for i, (lat, lon) in enumerate(discoveries):
            le, ln = to_mga.transform(lon, lat)
            lx, ly = ~src.transform * (le, ln)
            rx, ry = lx - (px_c - win_size//2), ly - (px_r - win_size//2)
            
            color = "lime" if i < 2 else "orange"
            draw.ellipse([rx-15, ry-15, rx+15, ry+15], outline=color, width=3)
            draw.text((rx+17, ry), f"DISCOVERY {i+1}", fill="white")

        im.save(OUT_PATH)
        print(f"DISCOVERY AUDIT SAVED TO: {OUT_PATH}")

if __name__ == "__main__":
    main()
