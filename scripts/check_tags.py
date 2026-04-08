import os, re
path = os.path.join('Test photos 3', 'DJI_0064.JPG')
with open(path, 'rb') as f: data = f.read()
s = data.find(b'<x:xmpmeta')
e = data.find(b'</x:xmpmeta>')
if s != -1 and e != -1:
    xstr = data[s:e+12].decode('utf-8', 'ignore')
    for m in re.findall(r'drone-dji:([A-Za-z]+)=\"([-0-9.,]+)\"', xstr):
        if 'lat' in m[0].lower() or 'lon' in m[0].lower() or 'gps' in m[0].lower() or 'alt' in m[0].lower():
            print(m)
