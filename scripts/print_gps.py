import json
d = json.load(open('gps_ground_truth.json'))
print(f"{'File':<20} {'Lat':>12} {'Lon':>13} {'Alt (m)':>10}")
print("-" * 58)
for k, v in d.items():
    print(f"{k:<20} {v['lat']:>12.7f} {v['lon']:>13.7f} {v['alt']:>10.1f}")
print(f"\nTotal: {len(d)} photos with GPS")
