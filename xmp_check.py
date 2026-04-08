import os

def check_xmp(image_path):
    with open(image_path, 'rb') as f:
        content = f.read()
        xmp_start = content.find(b'<x:xmpmeta')
        xmp_end = content.find(b'</x:xmpmeta>')
        if xmp_start != -1 and xmp_end != -1:
            return content[xmp_start:xmp_end+12].decode('utf-8', errors='ignore')
    return None

test_folder = r"C:\Users\True Debreuil\Documents\TERCOM\Test photos"
photos = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg'))]

for p in photos[:1]:
    full_path = os.path.join(test_folder, p)
    print(f"Checking XMP for: {p}")
    xmp = check_xmp(full_path)
    if xmp:
        # Looking for DJI specific tags
        for line in xmp.split('\n'):
            if 'drone-dji' in line or 'Gps' in line or 'RelativeAltitude' in line:
                print(line.strip())
    else:
        print("No XMP found.")
