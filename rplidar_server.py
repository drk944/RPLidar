from rplidar import RPLidar
import socket
import struct
import math
import time

# Create an RPLidar instance
# lidar = RPLidar('COM4')
lidar = RPLidar('/dev/ttyUSB1')
info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)
time.sleep(2)
# Create a UDP socket
server_address = ('localhost', 10000)
# server_address = ('192.168.0.112', 10000)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    # Collect Lidar scan data and send it over UDP
    for i, scan in enumerate(lidar.iter_scans(scan_type='express', max_buf_meas=False)):
        scan_data = b''.join(struct.pack('!ff', math.radians(measurement[1]), measurement[2]) for measurement in scan)
        sock.sendto(scan_data, server_address)
        length = len(scan_data)
        print(f"Sent scan {i} of length {len(scan)} with packet data size {length}") #: {scan_data}")

finally:
    # Close the Lidar connection when done
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
