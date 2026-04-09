import cv2
import numpy as np
import math
import rasterio
import os
from pyproj import Transformer

class PathVisualizer:
    def __init__(self, waypoints, dem_path="5m_DEM.tif"):
        self.canvas_size = 800
        self.waypoints = waypoints
        self.path_history = []
        self.trn_locks = []
        
        # Default dark background
        self.hillshade = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self.hillshade += 40 # Dark grey base
        
        # Load DEM for Background if available
        self.map_loaded = False
        if dem_path and os.path.exists(dem_path):
            try:
                print(f"[VIZ] Loading background elevation data from {dem_path}...")
                with rasterio.open(dem_path) as src:
                    self.src_crs = src.crs
                    self.src_transform = src.transform
                    self.full_width = src.width
                    self.full_height = src.height
                    
                    # Read 1st channel, resampled to 800x800
                    data_800 = src.read(1, out_shape=(self.canvas_size, self.canvas_size)).astype(np.float32)
                    
                    # Cleanup NoData
                    no_data_mask = data_800 < -1e10
                    if np.any(~no_data_mask):
                        data_800[no_data_mask] = np.nanmin(data_800[~no_data_mask])
                    else:
                        data_800[no_data_mask] = 0
                    
                    # Create shaded relief
                    dx = cv2.Sobel(data_800, cv2.CV_32F, 1, 0, ksize=3)
                    dy = cv2.Sobel(data_800, cv2.CV_32F, 0, 1, ksize=3)
                    
                    slope_mag = np.sqrt(np.clip(dx**2 + dy**2, 0, 1e6))
                    shade = 255 * (1.0 - (np.arctan(slope_mag * 0.2) / (np.pi/2)))
                    shade = np.clip(shade, 0, 255).astype(np.uint8)
                    
                    # Colorize for tactical look
                    color_map = cv2.cvtColor(shade, cv2.COLOR_GRAY2BGR)
                    overlay = np.zeros_like(color_map)
                    overlay[:] = (40, 20, 0)
                    self.hillshade = cv2.addWeighted(color_map, 0.5, overlay, 0.5, 0)
                    
                self.transformer = Transformer.from_crs("EPSG:4326", self.src_crs, always_xy=True)
                self.map_loaded = True
                print("[VIZ] DEM Background initialized successfully.")
            except Exception as e:
                print(f"[VIZ] WARNING: DEM background failed ({e}). Using standard grid.")
        
        if not self.map_loaded:
            self.lat_min, self.lat_max = -30.00, -29.98
            self.lon_min, self.lon_max = 153.22, 153.24

    def _gps_to_pixel(self, lat, lon):
        if self.map_loaded:
            try:
                # 1. GPS to MGA (Easting, Northing)
                easting, northing = self.transformer.transform(lon, lat)
                # 2. MGA to Full Map Pixel Coordinates
                col, row = ~self.src_transform * (easting, northing)
                # 3. Scale down to 800x800 Canvas
                x = int((col / self.full_width) * self.canvas_size)
                y = int((row / self.full_height) * self.canvas_size)
                return x, y
            except:
                pass
        
        # Fallback transform
        x = int((lon - self.lon_min) / (self.lon_max - self.lon_min) * self.canvas_size)
        y = int((self.lat_max - lat) / (self.lat_max - self.lat_min) * self.canvas_size)
        return x, y

    def update(self, drone_lat, drone_lon, state, match_type):
        # 1. Overlay on pre-rendered background
        img = self.hillshade.copy()
        
        # 2. Draw Grid (Subtle)
        grid_color = (100, 100, 100) if self.map_loaded else (150, 150, 150)
        for i in range(0, self.canvas_size, 100):
            cv2.line(img, (i, 0), (i, self.canvas_size), grid_color, 1)
            cv2.line(img, (0, i), (self.canvas_size, i), grid_color, 1)

        # 3. Draw Waypoints
        for i, wp in enumerate(self.waypoints):
            px, py = self._gps_to_pixel(wp[0], wp[1])
            cv2.circle(img, (px, py), 8, (0, 255, 255), -1) 
            cv2.putText(img, f"WP{i}", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 4. Draw Path History
        self.path_history.append((drone_lat, drone_lon))
        if len(self.path_history) > 1:
            points = np.array([self._gps_to_pixel(p[0], p[1]) for p in self.path_history], np.int32)
            cv2.polylines(img, [points], False, (0, 255, 0), 2) 

        # 5. Draw TRN Matches
        if "MATCH" in match_type:
            self.trn_locks.append((drone_lat, drone_lon))
        
        for lock in self.trn_locks:
            px, py = self._gps_to_pixel(lock[0], lock[1])
            cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 6, 2)

        # 6. Draw Drone
        dx, dy = self._gps_to_pixel(drone_lat, drone_lon)
        dx_clamped = np.clip(dx, 0, self.canvas_size-1)
        dy_clamped = np.clip(dy, 0, self.canvas_size-1)

        # Drone Marker
        cv2.circle(img, (dx_clamped, dy_clamped), 14, (0, 255, 0), 2)
        cv2.circle(img, (dx_clamped, dy_clamped), 6, (255, 255, 255), -1)
        cv2.putText(img, "TAILSITTER-01", (dx_clamped+20, dy_clamped+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 7. Information Panel (High contrast)
        cv2.rectangle(img, (0, 0), (320, 130), (20, 20, 20), -1)
        cv2.rectangle(img, (0, 0), (320, 130), (0, 255, 0), 2)
        cv2.putText(img, f"STATUS: {state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"LAT:   {drone_lat:.6f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"LON:   {drone_lon:.6f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        trn_color = (0, 255, 255) if "MATCH" in match_type else (100, 100, 100)
        cv2.putText(img, f"TRN:   {match_type}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trn_color, 2)
        
        # Save frame atomically to prevent viewer flicker
        tmp_name = "drone_path_viz_tmp.png"
        cv2.imwrite(tmp_name, img)
        try:
            os.replace(tmp_name, "drone_path_viz.png")
        except:
            pass
        
        return "drone_path_viz.png"
