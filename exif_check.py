import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None
        
        decoded = {}
        for tag, value in exif_data.items():
            decoded[TAGS.get(tag, tag)] = value
        return decoded
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def get_gps_info(exif_data):
    if 'GPSInfo' not in exif_data:
        return None
        
    gps_info = {}
    for key in exif_data['GPSInfo'].keys():
        decode = GPSTAGS.get(key, key)
        gps_info[decode] = exif_data['GPSInfo'][key]
    return gps_info

def to_float(val):
    try:
        return float(val)
    except:
        return 0.0

def convert_to_degrees(value):
    try:
        d = to_float(value[0])
        m = to_float(value[1])
        s = to_float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception as e:
        print(f"Angle conversion error: {e}")
        return 0.0

test_folder = r"C:\Users\True Debreuil\Documents\TERCOM\Test photos"
photos = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg'))]

for p in photos:
    full_path = os.path.join(test_folder, p)
    print(f"\n--- {p} ---")
    exif = get_exif_data(full_path)
    if exif:
        # Debug: Print some common tags
        # print(f"Make: {exif.get('Make')}, Model: {exif.get('Model')}")
        gps = get_gps_info(exif)
        if gps:
            try:
                lat = convert_to_degrees(gps['GPSLatitude'])
                if gps.get('GPSLatitudeRef') != 'N': lat = -lat
                lon = convert_to_degrees(gps['GPSLongitude'])
                if gps.get('GPSLongitudeRef') != 'E': lon = -lon
                alt = to_float(gps.get('GPSAltitude', 0))
                print(f"RESULT: Lat={lat:.8f}, Lon={lon:.8f}, Alt={alt:.2f}")
            except Exception as e:
                print(f"Error parsing GPS dict: {e}")
        else:
            print("No GPS tags.")
    else:
        print("No EXIF.")
