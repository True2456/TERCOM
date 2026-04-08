import os
path = os.path.join('Test photos 3', 'DJI_0064.JPG')
with open(path, 'rb') as f: data = f.read()
s = data.find(b'<x:xmpmeta')
e = data.find(b'</x:xmpmeta>')
if s != -1:
    xstr = data[s:e+12].decode('utf-8', 'ignore')
    with open('xmp_dump.txt', 'w') as out:
        out.write(xstr)
