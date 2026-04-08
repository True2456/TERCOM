import rasterio
from rasterio.windows import Window
import numpy as np
from pyproj import Transformer
import cv2

class TileManager:
    def __init__(self, tiff_path, target_res_x=512, target_res_y=512):
        self.src = rasterio.open(tiff_path)
        self.crs = self.src.crs
        self.transform = self.src.transform
        self.map_res = self.src.res[0] # Assumes square pixels (5m)
        self.target_res = (target_res_x, target_res_y)
        
        # Transformer for Lat/Lon to MGA
        self.transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        
        # Camera Params (OV9281 / 120-deg Diagonal)
        # Horizontal FOV ≈ 110 deg
        self.h_fov = 110.0
        
    def get_tile_at(self, lat, lon, alt_agl):
        """
        Extracts a map tile that matches the ground footprint of the camera.
        Uses altitude to determine the scale.
        """
        easting, northing = self.transformer.transform(lon, lat)
        
        # Calculate ground footprint width: 2 * alt * tan(FOV/2)
        footprint_w = 2 * alt_agl * np.tan(np.deg2rad(self.h_fov / 2.0))
        
        # Map pixels needed for this footprint
        pixels_needed = footprint_w / self.map_res
        
        # Window center in map coordinates
        row, col = ~self.transform * (easting, northing)
        
        # Buffer the window to handle rotation later
        window_size = int(pixels_needed * 1.5)
        
        if window_size < 1:
            return None # Avoid empty window
        
        win = Window(
            col - window_size // 2,
            row - window_size // 2,
            window_size,
            window_size
        )
        
        # Read the window (memory efficient)
        tile = self.src.read(1, window=win, boundless=True, fill_value=0)
        
        if tile.size == 0 or tile.shape[0] == 0 or tile.shape[1] == 0:
            return None
            
        # Pre-scaling: Resize map tile to match expected visual scale
        # If camera resolution is 1280x800, we might match at a lower res for speed
        scaled_tile = cv2.resize(tile, self.target_res, interpolation=cv2.INTER_LINEAR)
        
        # Normalize/Hillshade (since it's a DEM)
        # Simple gradient-based hillshade for matching
        dy, dx = np.gradient(scaled_tile.astype(np.float64))
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        
        # Return a normalized 0-255 grayscale
        norm_tile = cv2.normalize(slope, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return norm_tile

    def close(self):
        self.src.close()

if __name__ == "__main__":
    # Test on PC first
    tm = TileManager('5m_DEM.tif')
    # Use center coords from before
    lat, lon = -29.98, 153.17 # Approx
    tile = tm.get_tile_at(lat, lon, 100) # 100m AGL
    cv2.imwrite('map_tile_sample.png', tile)
    print("Map tile sample saved.")
    tm.close()
