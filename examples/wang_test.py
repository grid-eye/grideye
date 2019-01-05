import time
import busio
import board
import adafruit_amg88xx
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
points = [(math.floor(i/8) , i%8) for i in range(0,64)]
grid_x ,grid_y = np.mgrid[0:7:32j,0:7:32j]

i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
discard_frames = 2
for i in range(discard_frames):
    for row in amg.pixels:
        pass
plt.ion()
(fig,axis)=plt.subplots()
while True:
    allpixels=[]
    for row in amg.pixels:
        # Pad to 1 decimal place
       # print(['{0:.1f}'.format(temp) for temp in row])
        #print("")
        allpixels.append(row)
#    print("\n")
   # print(np.array(allpixels).shape)
    inter_result = griddata(points,np.array(allpixels).flatten(),(grid_x,grid_y),method='cubic')
   # print(inter_result.shape)
    axis.imshow(inter_result)
    plt.draw()
    plt.pause(0.00001)
