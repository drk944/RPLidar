from rplidar import RPLidar
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import threading


def draw():
    global is_plot
    while is_plot:
        plt.figure(1)
        plt.cla()
        plt.ylim(-9000, 9000)
        plt.xlim(-9000, 9000)
        plt.scatter(x, y, c='r', s=8)
        plt.pause(0.0001)
    plt.close("all")

# def save_data():
#     global is_save
#     while is_save:

#         np.savetxt('scans.txt', np.c_[x, y], delimiter=',')
#         time.sleep(1)


is_plot = True
is_save = True

x = []
y = []

# lidar = RPLidar('COM4')
lidar = RPLidar('/dev/ttyUSB0')
info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)

time.sleep(2)


# threading.Thread(target=draw).start()
# threading.Thread(target=save_data).start()

# lidar.iter_scans = 3400
scans = []
# counter = 0
for i, scan in enumerate(lidar.iter_scans(scan_type='express', max_buf_meas=False)):
    print('%d: Got %d measurments' % (i, len(scan)))
    if i > 10:
        break
    x = np.zeros(len(scan))
    y = np.zeros(len(scan))

    for j in range(len(scan)):
        x[j] = scan[j][2] * math.cos(math.radians(scan[j][1]))
        y[j] = scan[j][2] * math.sin(math.radians(scan[j][1]))
    scans.append([x, y])

scans = np.array(scans)
np.save('bd2', scans)
lidar.stop()
lidar.stop_motor()
lidar.disconnect()
