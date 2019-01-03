import time
import busio
import board
import adafruit_amg88xx
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
frame_x = 8
frame_y = 8
# we need th_bgframes frames  to calculate the average temperature of the bground
th_bgframes = 30
# the counter of the bgframes frames
bgframe_cnt = 0
all_bgframes = []
pre_read_count = 2
# 8X8 grid

points = [(math.floor(ix/8), (ix % 8)) for ix in range(0, 64)]
grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]
for i in range(pre_read_count):
    for row in amg.pixels:
        pass


# 插值(cublic)
def cubicInterpolate(points, pixels, grid_x, grid_y, ip_type='cubic'):
    return griddata(points, pixels, (grid_x, grid_y), method=ip_type)


while True:
    temp = []
    for row in amg.pixels:
        # Pad to 1 decimal place
        temp.append(row)
    bgframe_cnt = bgframe_cnt + 1
    all_bgframes.append(temp)
    print('next step')
    print(len(all_bgframes))
    total_frames = np.zeros((8, 8))
    # the result of the interpolating for the grid
    average_temp = np.array(average_temperature).flatten(), grid_x, grid_y, 'cubic')
    temp=cubicInterpolate(points, np.array(temp).flatten()
                , grid_x, grid_y, 'cubic')
    print("after interpolating , save the data in file:average_temperature.npy")
    fig, (ax1, ax2, ax3)=plt.subplots(3, 1)
    ax1.imshow(inter_result, cmap = 'hot', interpolation = 'bicubic')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis([0, 31, 32, 0])
    ax2.hist(np.array(inter_result).ravel(), bins = 256,
             range = (19, 28), fc = 'k', ec = 'k')
    ax2.set_xlabel('temperature')
    ax2.set_ylabel('%')
    temp_diff=np.array(temp) - inter_result
    ax3.hist(temp_diff.ravel(), bins = 256,
             range = (-1.0, 0.9), fc = 'k', ec = 'k')
    ax3.set_xlabel('temperature-diff')
    ax3.set_ylabel('%')
    plt.show()
    break
