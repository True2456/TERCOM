"""
Read GPS EXIF from all JPGs in 'Test photos 2'.
READ-ONLY — files are never modified.
"""
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os, json

def get_gps(img_path):
    """Extract GPS coords from EXIF. Returns (lat, lon, alt_m) or None."""
    img = Image.open(img_path)
    exif = img._getexif()
    if not exif:
        return None
    gps_info = {}
    for tag, val in exif.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == 'GPSInfo':
            for k, v in val.items():
                gps_info[GPSTAGS.get(k, k)] = v
    if not gps_info:
        return None
    def to_deg(vals, ref):
        d, m, s = float(vals[0]), float(vals[1]), float(vals[2])
        deg = d + m/60 + s/3600
        if ref in ['S', 'W']:
            deg = -deg
        return deg
    try:
        lat = to_deg(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
        lon = to_deg(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
        alt = float(gps_info.get('GPSAltitude', 0))
        return lat, lon, alt
    except KeyError:
        return None

folder = 'Test photos 2'
jpgs = sorted([f for f in os.listdir(folder) if f.endswith('.JPG')])
print(f"Found {len(jpgs)} JPGs\n")
print(f"{'File':<20} {'Lat':>12} {'Lon':>13} {'Alt (m)':>10}")
print("-" * 60)

results = {}
for f in jpgs:
    path = os.path.join(folder, f)
    gps = get_gps(path)
    if gps:
        lat, lon, alt = gps
        print(f"{f:<20} {lat:>12.7f} {lon:>13.7f} {alt:>10.1f}")
        results[f] = {"lat": lat, "lon": lon, "alt": alt}
    else:
        print(f"{f:<20} NO GPS DATA")

# Save to JSON for use by the matching script
with open("gps_ground_truth.json", "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nSaved ground truth to gps_ground_truth.json")
