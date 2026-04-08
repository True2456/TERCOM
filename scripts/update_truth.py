import os, json
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

truth_file = 'gps_ground_truth.json'
with open(truth_file) as f:
    truth = json.load(f)

def get_decimal(tup, ref):
    if not tup: return 0.0
    d = float(tup[0])
    m = float(tup[1])
    s = float(tup[2])
    deg = d + (m / 60.0) + (s / 3600.0)
    if ref in ['S', 'W']:
        deg = -deg
    return deg

for f in os.listdir('Test photos 3'):
    if not f.endswith('.JPG'): continue
    path = os.path.join('Test photos 3', f)
    try:
        img = Image.open(path)
        exif = img.getexif()
        if exif:
            gps_info = {}
            for key, value in exif.get_ifd(0x8825).items():
                gps_info[GPSTAGS.get(key, key)] = value
            
            lat = get_decimal(gps_info.get('GPSLatitude'), gps_info.get('GPSLatitudeRef'))
            lon = get_decimal(gps_info.get('GPSLongitude'), gps_info.get('GPSLongitudeRef'))
            alt = float(gps_info.get('GPSAltitude', 150))
            if lat != 0.0 and lon != 0.0:
                truth[f] = {'lat': lat, 'lon': lon, 'alt': alt}
    except Exception as e:
        print(f"Error reading {f}: {e}")

with open(truth_file, 'w') as f:
    json.dump(truth, f, indent=2)
print('Updated gps_ground_truth.json for Test photos 3')
