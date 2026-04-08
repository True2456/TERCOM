import socket
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 14551

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on {UDP_IP}:{UDP_PORT} for 10 seconds...")
sock.settimeout(10.0)

start_time = time.time()
packet_count = 0

try:
    while time.time() - start_time < 10:
        try:
            data, addr = sock.recvfrom(1024)
            packet_count += 1
            print(f"[{packet_count}] Received {len(data)} bytes from {addr}")
            # Print hex for first 10 bytes to identify MAVLink start (0xFE or 0xFD)
            print(f"   Hex: {data[:10].hex()}")
        except socket.timeout:
            continue
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()
    print(f"Total packets received: {packet_count}")
