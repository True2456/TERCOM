import cv2, os, json, math
import numpy as np
import rasterio
from pyproj import Transformer

with open('gps_ground_truth.json', 'r') as f: truth_db = json.load(f)
gps = truth_db['DJI_0083.JPG']
gps_lat, gps_lon = gps['lat'], gps['lon']

def footprint_px(agl_m, hfov_deg, img_w, img_h, ortho_res):
    fov_w_m = 2.0 * agl_m * math.tan(math.radians(hfov_deg / 2.0))
    return int(fov_w_m / ortho_res), int((fov_w_m * (img_h / img_w)) / ortho_res)

img_gray = cv2.imread(os.path.join('Test photos 3', 'DJI_0083.JPG'), cv2.IMREAD_GRAYSCALE)
h, w = img_gray.shape[:2]
tile_w, tile_h = footprint_px(150, 69.7, w, h, 0.5)

src = rasterio.open(r'C:\Users\True Debreuil\Documents\RedRock Pi color 1 res.tif')
to_mga = Transformer.from_crs('EPSG:4326', 'EPSG:28356', always_xy=True)
gps_e, gps_n = to_mga.transform(gps_lon, gps_lat)
c, r = ~src.transform * (gps_e, gps_n)
search_w = search_h = tile_w + 400
win = rasterio.windows.Window(int(c) - search_w//2, int(r) - search_h//2, search_w, search_h)
patch = src.read(window=win)
ortho_gray = cv2.cvtColor(np.transpose(patch, (1,2,0)), cv2.COLOR_RGBA2GRAY)

o_edge = cv2.dilate(cv2.Canny(cv2.GaussianBlur(ortho_gray, (5, 5), 0), 50, 150), np.ones((3,3), np.uint8), iterations=1)

yaws_to_test = [106.3, -106.3, 106.3-90, 106.3+90, -106.3-90, -106.3+90, 0, 180, 106.3+180, -106.3+180]
for a in yaws_to_test:
    M = cv2.getRotationMatrix2D((w//2, h//2), a, 1.0)
    rot = cv2.warpAffine(img_gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    res = cv2.resize(rot, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    p_edge = cv2.dilate(cv2.Canny(cv2.GaussianBlur(res, (5, 5), 0), 50, 150), np.ones((3,3), np.uint8), iterations=1)
    
    score = cv2.minMaxLoc(cv2.matchTemplate(o_edge, p_edge, cv2.TM_CCOEFF_NORMED))[1]
    print(f'Angle {a:7.1f}: Score = {score:.3f}')
