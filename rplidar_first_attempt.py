from rplidar import RPLidar
import time
lidar = RPLidar('/dev/ttyUSB0')

info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)

time.sleep(1)
# lidar.stop_motor()
# time.sleep(5)
# lidar.start_motor()

# ob = lidar.iter_scans(scan_type='express')
# print(ob)

for i, scan in enumerate(lidar.iter_scans()):
    print('%d: Got %d measurments' % (i, len(scan)))
    if i > 10:
        break

lidar.stop()
lidar.stop_motor()
lidar.disconnect()