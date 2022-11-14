'''
This script allows for easy point cloud logging.
Press space to record a point cloud
Press Escape to save and exit
'''
from rplidar import RPLidar
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import threading
import os

# Windows
if os.name == 'nt':
    import msvcrt

# Posix (Linux, OS X)
else:
    import sys
    import termios
    import atexit
    from select import select


class KBHit:

    def __init__(self):
        '''Creates a KBHit object that you can call to do various keyboard things.'''
        if os.name == 'nt':
            pass
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)

    def set_normal_term(self):
        ''' Resets to normal terminal.  On Windows this is a no-op.'''
        if os.name == 'nt':
            pass
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def getch(self):
        ''' Returns a keyboard character after kbhit() has been called.
            Should not be called in the same program as getarrow().'''
        s = ''
        if os.name == 'nt':
            return msvcrt.getch().decode('utf-8')
        else:
            return sys.stdin.read(1)

    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.'''
        if os.name == 'nt':
            return msvcrt.kbhit()
        else:
            dr,dw,de = select([sys.stdin], [], [], 0)
            return dr != []

'''
# Test    
if __name__ == "__main__":
    kb = KBHit()
    print('Hit any key, or ESC to exit')
    while True:
        if kb.kbhit():
            c = kb.getch()
            if ord(c) == 27: # ESC
                break
            print("Key:", c)
    kb.set_normal_term()
'''

def draw():
    global is_plot
    while is_plot:
        plt.figure(1)
        plt.cla()
        plt.ylim(-9000, 9000)
        plt.xlim(-9000, 9000)
        try:
            plt.scatter(x, y, c='r', s=8)
            plt.pause(0.0001)
        except:
            pass
    plt.close("all")

kb = KBHit()
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

scans = []
# counter = 0

# threading.Thread(target=draw).start()

for i, scan in enumerate(lidar.iter_scans(scan_type='express', max_buf_meas=False)):
    # print('%d: Got %d measurments' % (i, len(scan)))
    # if i > 10:
        # break
    x = np.zeros(len(scan))
    y = np.zeros(len(scan))

    for j in range(len(scan)):
        x[j] = scan[j][2] * math.cos(math.radians(scan[j][1]))
        y[j] = scan[j][2] * math.sin(math.radians(scan[j][1]))

    if kb.kbhit():
        c = kb.getch()
        if ord(c) == 27: # ESC
            break
            print("Key:", c)
        if ord(c) == 32:
            scans.append([x, y])
            print("Scan Saved")

kb.set_normal_term()
scans = np.array(scans, dtype=object)
np.save('test_track_5', scans)
# notes:
# test_cb460_1: iterative scans at 10cm. 3m forwards, 2m right
# test_cb460_2: 5 scans moving around a square clockwise 50cm spacing. Should end up at start
# test_cb460_3: 5 scans around the same square. Starting at 45 degrees, rotating clockwise 45 degrees at each point. Should end up back at start 180 degrees from original starting point.

# test_track_1: 4 scans 10 cm apart moving straight forward
# test_track_2: 5 scans 10 cm apart moving to the right
# test_track_3: 2 scans. translation 10cm forward then 45 degrees clockwise rotation
# test_track_4: 5 scans, same place, rotated 45 degrees each time clockwise
# test_track_5: 2 scans, 45 degrees clockwise rotation, 10cm forward, 

lidar.stop()
lidar.stop_motor()
lidar.disconnect()
