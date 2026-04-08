import os
import json
import random
import numpy as np
import cv2
from tqdm import tqdm
from simulator.map_renderer import MapRenderer
from pyproj import Transformer

def generate_dataset(tiff_path, output_dir, num_frames=5000):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
    
    renderer = MapRenderer(tiff_path)
    
    # Get bounds for random sampling
    # EPSG:7856 Bounds: left=497990, bottom=6658046, right=534185, top=6707906
    # Buffer to stay inside
    buffer = 5000 
    
    to_wgs84 = Transformer.from_crs(renderer.crs, "EPSG:4326", always_xy=True)
    
    metadata = {}
    
    print(f"Generating {num_frames} frames...")
    
    for i in tqdm(range(num_frames)):
        # Random position
        easting = random.uniform(497990 + buffer, 534185 - buffer)
        northing = random.uniform(6658046 + buffer, 6707906 - buffer)
        
        lon, lat = to_wgs84.transform(easting, northing)
        
        # Random flight parameters
        alt_agl = random.uniform(60, 150) # 60m to 150m AGL
        pitch = random.uniform(-10, 10)
        roll = random.uniform(-10, 10)
        yaw = random.uniform(0, 360)
        
        # Random lighting
        sun_az = random.uniform(0, 360)
        sun_el = random.uniform(30, 80) # Avoid very low sun to prevent extreme shadows
        
        frame = renderer.render(lat, lon, alt_agl, pitch, roll, yaw, sun_az, sun_el)
        
        if frame is not None:
            filename = f"frame_{i:05d}.png"
            cv2.imwrite(os.path.join(output_dir, 'frames', filename), frame)
            
            metadata[filename] = {
                "lat": lat,
                "lon": lon,
                "alt_agl": alt_agl,
                "pitch": pitch,
                "roll": roll,
                "yaw": yaw,
                "sun_az": sun_az,
                "sun_el": sun_el,
                "mga": [easting, northing]
            }
            
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Dataset generated in {output_dir}")

if __name__ == "__main__":
    generate_dataset('5m_DEM.tif', 'dataset_v1', num_frames=50) # Small batch for testing
