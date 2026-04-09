import os, re
for f in os.listdir('Test photos 3'):
    if not f.endswith('.JPG'): continue
    path = os.path.join('Test photos 3', f)
    with open(path, 'rb') as fp: data = fp.read()
    s = data.find(b'<x:xmpmeta')
    e = data.find(b'</x:xmpmeta>')
    if s != -1:
        xstr = data[s:e+12].decode('utf-8', 'ignore')
        yaw = re.search(r'drone-dji:FlightYawDegree=\"([-0-9.]+)\"', xstr)
        print(f'{f}: {yaw.group(1) if yaw else "Not Found"}')
