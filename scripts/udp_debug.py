import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 14551

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on {UDP_IP}:{UDP_PORT}...")
sock.settimeout(5.0)

try:
    data, addr = sock.recvfrom(1024)
    print(f"Received {len(data)} bytes from {addr}")
except socket.timeout:
    print("Timed out. No packets received.")
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()
