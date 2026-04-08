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
        if exif_data:
            for tag, value in exif_data.items():
                decoded[TAGS.get(tag, tag)] = value
        return decoded
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

test_folder = r"C:\Users\True Debreuil\Documents\TERCOM\Test photos"
photos = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg'))]

for p in photos[:1]:
    full_path = os.path.join(test_folder, p)
    print(f"FULL EXIF TAGS FOR: {p}")
    exif = get_exif_data(full_path)
    if exif:
        for k, v in exif.items():
            if k == 'GPSInfo':
                print("GPSInfo found, keys:", v.keys())
                for gk, gv in v.items():
                    print(f"  {GPSTAGS.get(gk, gk)}: {gv}")
            else:
                # Print non-binary tags
                if not isinstance(v, bytes):
                    print(f"{k}: {v}")
    else:
        print("No EXIF.")
