import sys, re
filename = r'Test photos 2/DJI_0041.JPG'
with open(filename, 'rb') as f:
    data = f.read()
xmp_start = data.find(b'<x:xmpmeta')
xmp_end = data.find(b'</x:xmpmeta>')
if xmp_start != -1 and xmp_end != -1:
    xmp_str = data[xmp_start:xmp_end + 12].decode('utf-8', errors='ignore')
    matches = re.findall(r'drone-dji:(.*?)=\"(.*?)\"', xmp_str)
    for k, v in matches:
        if 'Altitude' in k or 'Focal' in k or 'FOV' in k or 'Resolution' in k:
            print(f'{k}: {v}')
