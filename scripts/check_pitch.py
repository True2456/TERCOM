import os, re
import json

with open("gps_ground_truth.json") as f:
    gps_truth = json.load(f)

for photo_name in sorted([k for k in gps_truth if k.endswith(".JPG")]):
    with open(os.path.join('Test photos 2', photo_name), 'rb') as f:
        data = f.read()
    s = data.find(b'<x:xmpmeta')
    e = data.find(b'</x:xmpmeta>')
    if s!=-1 and e!=-1:
        xstr = data[s:e+12].decode('utf-8', 'ignore')
        yaw = pitch = "N/A"
        y_m = re.search(r'drone-dji:GimbalYawDegree=\"([-0-9.]+)\"', xstr)
        p_m = re.search(r'drone-dji:GimbalPitchDegree=\"([-0-9.]+)\"', xstr)
        fp_m = re.search(r'drone-dji:FlightPitchDegree=\"([-0-9.]+)\"', xstr)
        if y_m: yaw = float(y_m.group(1))
        if p_m: pitch = float(p_m.group(1))
        
        print(f"{photo_name}: GimbalPitch={pitch}  FlightPitch={fp_m.group(1) if fp_m else 'N/A'}")
