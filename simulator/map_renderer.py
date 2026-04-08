import os
import numpy as np
import rasterio
from pyproj import Transformer
import cv2

class MapRenderer:
    def __init__(self, tiff_path):
        print("Initializing MapRenderer (Lightweight Numpy Mode)...")
        # Load GeoTIFF
        with rasterio.open(tiff_path) as src:
            print(f"Reading {tiff_path}...")
            self.crs = src.crs
            self.transform = src.transform
            self.bounds = src.bounds
            self.res = src.res
            self.data = src.read(1).astype(np.float32)
            
        # Handle "No Data" values (-3.4028e+38)
        no_data_mask = self.data < -1e10
        self.data[no_data_mask] = np.nanmin(self.data[~no_data_mask])
        
        self.height, self.width = self.data.shape
        print(f"Map size: {self.width}x{self.height}")
        
        # Projection Transformer (assuming WGS84 input for lat/lon)
        self.transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        
        # Camera Intrinsics (OV9281 120-deg FOV Diagonal)
        self.res_x, self.res_y = 1280, 800
        h_fov = 110.0 # Approximate
        v_fov = 85.0  # Approximate
        
        self.fx = (self.res_x / 2.0) / np.tan(np.deg2rad(h_fov / 2.0))
        self.fy = (self.res_y / 2.0) / np.tan(np.deg2rad(v_fov / 2.0))
        self.cx, self.cy = self.res_x / 2.0, self.res_y / 2.0
        
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def latlon_to_mga(self, lat, lon):
        return self.transformer.transform(lon, lat)

    def render(self, lat, lon, alt_agl, pitch=0, roll=0, yaw=0, sun_azimuth=0, sun_elevation=45):
        """
        Renders a nadir-view grayscale frame from the DEM using numpy.
        """
        try:
            easting, northing = self.latlon_to_mga(lat, lon)
            
            # Find map indices
            row, col = ~self.transform * (easting, northing)
            row, col = int(row), int(col)
            
            # Crop size (oversample to allow rotation)
            crop_size = int(max(self.res_x, self.res_y) * 2.0) # More buffer for rotation
            
            r_start, r_end = row - crop_size//2, row + crop_size//2
            c_start, c_end = col - crop_size//2, col + crop_size//2
            
            # Boundary checks
            if r_start < 0 or c_start < 0 or r_end >= self.height or c_end >= self.width:
                return None 
                
            crop = self.data[r_start:r_end, c_start:c_end]
            
            # Hillshading Logic
            dz_dx, dz_dy = self._compute_gradients(crop)
            hillshade = self._calculate_hillshade(dz_dx, dz_dy, sun_azimuth, sun_elevation)
            
            # Perspective/Rotation Warp
            frame = self._warp_and_crop(hillshade, yaw)
            
            # Add Noise
            frame = self._add_noise(frame)
            
            return frame
        except Exception as e:
            print(f"[MapRenderer] Render error: {e}")
            return None

    def _compute_gradients(self, array):
        # Using Sobel for stable gradients
        # res[0] is pixel width, res[1] is pixel height
        dx = cv2.Sobel(array, cv2.CV_32F, 1, 0, ksize=3) / (8 * self.res[0])
        dy = cv2.Sobel(array, cv2.CV_32F, 0, 1, ksize=3) / (8 * self.res[1])
        return dx, dy

    def _calculate_hillshade(self, dz_dx, dz_dy, azimuth, elevation):
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)
        
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect = np.arctan2(dz_dx, -dz_dy)
        
        shaded = (np.sin(el_rad) * np.cos(slope)) + \
                 (np.cos(el_rad) * np.sin(slope) * np.cos(az_rad - aspect))
                 
        return np.clip(shaded, 0, 1)

    def _warp_and_crop(self, hillshade, yaw):
        h, w = hillshade.shape
        img = (hillshade * 255).astype(np.uint8)
        
        # Simple rotation for now (yaw)
        M = cv2.getRotationMatrix2D((w/2, h/2), yaw, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # Crop to camera resolution
        start_y = (h - self.res_y) // 2
        start_x = (w - self.res_x) // 2
        final = rotated[start_y:start_y+self.res_y, start_x:start_x+self.res_x]
        
        return final

    def _add_noise(self, frame):
        noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    renderer = MapRenderer('5m_DEM.tif')
    # Sample render (center of bounds)
    # Bounds: left=497990, bottom=6658046, right=534185, top=6707906
    # Easting approx 516000, Northing approx 6682000
    # need to find lat/lon for these
    from pyproj import Transformer
    to_wgs84 = Transformer.from_crs(renderer.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(516000, 6682000)
    
    frame = renderer.render(lat, lon, 100) # 100m AGL
    if frame is not None:
        cv2.imwrite('sample_render.png', frame)
        print("Sample render saved to sample_render.png")
