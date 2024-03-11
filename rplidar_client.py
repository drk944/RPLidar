import socket
import struct
import matplotlib.pyplot as plt
import numpy as np
import math

angle_offset_deg = -15  # Offset angle in degrees
angle_offset_rad = math.radians(angle_offset_deg)  # Convert the offset angle to radians
# Create a UDP socket
server_address = ('', 10000)  # Use the IP address of the Raspberry Pi
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(server_address)

# Create a figure and axis for the plot
fig, ax = plt.subplots()
scatter = ax.scatter([], [], c='r', s=8)  # Initialize scatter plot

# Define the update function to be called by FuncAnimation
def update_plot(scan):
    scan_len = len(scan) // 2
    scan = np.array(scan).reshape(scan_len, 2)
    x_data = np.zeros(scan_len)
    y_data = np.zeros(scan_len)
    print(f"scan of length {scan_len}")
    # reshape the scan data
    # Update x and y coordinates from the scan data
    for i in range(scan_len):
        angle = scan[i][0] + angle_offset_rad

        y_data[i] = (scan[i][1]/1000 * math.cos(angle))
        x_data[i] = (scan[i][1]/1000 * math.sin(angle))
        
    scatter.set_offsets(np.column_stack((x_data, y_data)))  # Update scatter plot data
    # Draw the robot coordinate frame axes
    ax.plot([0, 0.5], [0, 0], 'red')  # X-axis
    ax.plot([0, 0], [0, 0.5], 'green')  # Y-axis
    return scatter,

try:
    # Listen for UDP packets and update the plot
    while True:
        data, _ = sock.recvfrom(2048+1024)
        num_measurements = len(data) // 8  # Each measurement consists of two floats (8 bytes total)
        scan = struct.unpack('!{}f'.format(num_measurements * 2), data)
        update_plot(scan)
        
        ax.set_xlim(-7, 7)  # Set the x-axis limits
        ax.set_ylim(-7, 7)  # Set the y-axis limits
        ax.set_aspect('equal')  # Set the aspect ratio of the plot to be equal
        ax.grid(True)  # Turn on the grid
        # set background color to grey
        ax.set_facecolor('lightgrey')
        plt.pause(0.001)
finally:
    # Close the UDP socket
    sock.close()
