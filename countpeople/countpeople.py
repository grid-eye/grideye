import numpy as np
import time
import busio
import board
import adafruit_amg88xx
import math
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
me_x = 8
frame_y = 8
# we need th_bgframes frames  to calculate the average temperature of th$
th_bgframes = 30
# the counter of the bgframes frames
bgframe_cnt = 0
all_bgframes= []
pre_read_count = 2
# 8X8 grid

points = [(math.floor(ix/8),(ix%8)) for ix  in range(0,64)]
# me_x = 8
frame_y = 8
# we need th_bgframes frames  to calculate the average temperature of th$
th_bgframes = 30
# the counter of the bgframes frames
bgframe_cnt = 0
all_bgframes= []
pre_read_count = 2
# 8X8 grid

points = [(math.floor(ix/8),(ix%8)) for ix  in range(0,64)]
grid_x,grid_y = np.mgrid[0:7:32j,0:7:32j]
for i in range(pre_read_count):
        for row in amg.pixels:
                    pass

grid_x,grid_y = np.mgrid[0:7:32j,0:7:32j]
#discard the first and the second frame
for i in range(pre_read_count):
        for row in amg.pixels:
                    pass

class CountPeople:
    def __init__(self,pre_read_count=2,th_bgframes=20):
        #the counter of the bgframes
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.amg = adafruit_amg88xx.AMG88XX(i2c)
        self.grid_x,self.grid_y=np.mgrid[0:7:32j,0:7:32j]
        self.bgframe_cnt =0
        self.all_bgframes=[] #save all frames which sensor read
        self.pre_read_count=pre_read_count
        self.th_bgframes = th_bgframes
        #8*8 grid
        self.points = [(math.floor(ix/8),(ix%8)) for ix  in range(0,64)]
        #discard the first and the second frame
        for i in range(self.pre_read_count):
            for row in self.amg.pixels:
                pass
    def interpolate(self, points ,pixels,grid_x,grid_y,inter_type='cubic'):
        '''
        interpolating for the pixels,default method is cubic
        '''
        return griddata(points,pixels,(grid_x,grid_y),method=inter_type)
    def readPixelsArray(self):
        '''
          func:  read pixels
          args:none
          return :2-d array
        '''
        pass
    def calAverageTemp(self):
        '''
           func: calulate the temperature of n frames ,n >= 200
            args:none
            return : 2-d array which is the average temperature of
                    n frames
        '''
        if len(self.all_bgframes) <(self.th_bgframes):
            raise RuntimeError('the len of the all_bgframes is too small')
        total_frames=np.zeros((8,8))
        for aitem in range(len(self.all_bgframes)):
            total_frames = total_frames+np.array(self.all_bgframes[aitem])
        return total_frames/self.th_bgframes
    def detectionNoise(self):
        '''
            func:remove noise from image
            args:none
            return :2-d array
        '''
        pass
    def process(self):
        '''
            main function
        '''
        while True:
            temp = []
            for row in self.amg.pixels:
            #Pad to 1 decimal place
                temp.append(row)
            self.bgframe_cnt=self.bgframe_cnt+1
            self.all_bgframes.append(temp)
            if self.bgframe_cnt >= self.th_bgframes:
                average_temperature=self.calAverageTemp()
                # the result of the interpolating for the grid
                average_temp = np.array(average_temperature).flatten()
                inter_result = self.interpolate(points,average_temp,grid_x,grid_y,'cubic')
                np.save('/tmp/average_temperature.npy',inter_result)
                print(average_temp)
                print("after save")
                break
    def  extractBody(self):
        pass
    def findBodyLocation(self):
        pass
    def trackPeople(self):
        pass
countp = CountPeople()
countp.process()
