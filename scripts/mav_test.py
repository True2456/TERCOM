from pymavlink import mavutil
import time

print("Connecting to udpin:127.0.0.1:14551...")
mav = mavutil.mavlink_connection('udpin:127.0.0.1:14551')

print("Waiting for heartbeat...")
hb = mav.wait_heartbeat(timeout=10)
if hb:
    print(f"Heartbeat received from System {mav.target_system}, Component {mav.target_component}")
else:
    print("Heartbeat timeout!")
