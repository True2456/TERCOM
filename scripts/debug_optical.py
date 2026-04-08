import cv2, os, rasterio, math, numpy as np
from pyproj import Transformer
import re

photo = 'Test photos 2/DJI_0041.JPG'
ORTHO_RES = 2.0; MARGIN_M = 100.0

# Drone limits footprint
img_gray = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)
gnd_w = 2 * 150.0 * math.tan(math.radians(82.1 / 2))
gnd_h = gnd_w * (img_gray.shape[0] / img_gray.shape[1])
tile_w, tile_h = max(1, int(gnd_w / ORTHO_RES)), max(1, int(gnd_h / ORTHO_RES))

# The resized drone photo the algorithm uses:
photo_resized = cv2.resize(img_gray, (tile_w, tile_h))

# Ortho patch logic
gps_lat, gps_lon = -29.9823237, 153.2264899
margin_px = int(MARGIN_M / ORTHO_RES)
search_w = tile_w + 2 * margin_px
search_h = tile_h + 2 * margin_px
diag = int(math.ceil(math.sqrt(search_w**2 + search_h**2)))

with rasterio.open(r'C:\Users\True Debreuil\Documents\redrockv2.tif') as src:
    to_mga = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
    e, n = to_mga.transform(gps_lon, gps_lat)
    col, row = ~src.transform * (e, n)
    win = rasterio.windows.Window(int(col) - diag//2, int(row) - diag//2, diag, diag)
    o_mb = src.read(window=win)
    o_rgb = cv2.cvtColor(np.transpose(o_mb, (1,2,0)), cv2.COLOR_RGBA2GRAY)
    
# Rotate
def get_yaw(p):
    with open(p, 'rb') as f:
        data = f.read()
    s = data.find(b'<x:xmpmeta'); e = data.find(b'</x:xmpmeta>')
    if s!=-1 and e!=-1:
        xstr = data[s:e+12].decode('utf-8', errors='ignore')
        m = re.search(r'drone-dji:FlightYawDegree=\"([-0-9.]+)\"', xstr)
        if m: return float(m.group(1))
    return 0.0
yaw = get_yaw(photo)
M = cv2.getRotationMatrix2D((diag//2, diag//2), yaw, 1.0)
rotated = cv2.warpAffine(o_rgb, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
crop_x = diag//2 - search_w//2
crop_y = diag//2 - search_h//2
map_patch = rotated[crop_y:crop_y+search_h, crop_x:crop_x+search_w]

# Canny filter check to emulate the algorithm
p_c = cv2.Canny(photo_resized, 30, 150)
m_c = cv2.Canny(map_patch, 30, 150)

# Build a visual
out = np.zeros((search_h, search_w + tile_w + 10, 3), dtype=np.uint8)

# Grayscale view
out_gray = out.copy()
out_gray[margin_px:margin_px+tile_h, :tile_w, 0] = photo_resized
out_gray[margin_px:margin_px+tile_h, :tile_w, 1] = photo_resized
out_gray[margin_px:margin_px+tile_h, :tile_w, 2] = photo_resized
out_gray[:, tile_w+10:, 0] = map_patch
out_gray[:, tile_w+10:, 1] = map_patch
out_gray[:, tile_w+10:, 2] = map_patch

# Scaling up the final image 4x so it is visible to humans (nearest neighbor to avoid artificial blur)
cv2.imwrite('optical_debug_gray.png', cv2.resize(out_gray, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))

# Edge view
out[margin_px:margin_px+tile_h, :tile_w, 1] = p_c
out[:, tile_w+10:, 2] = m_c
cv2.imwrite('optical_debug_edge.png', cv2.resize(out, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))

print('Saved optical_debug_gray.png and optical_debug_edge.png')
