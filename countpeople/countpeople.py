import numpy as np
import cv2 as cv
import time
import busio
import board
import adafruit_amg88xx
import math
import scipy
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# sleep 10 s
if len(sys.argv) > 1:
    if sys.argv[1] == 'people':
        print('test human,sleep 10 s')
        time.sleep(10)
else:
    time.sleep(2)
# sys.argv[2] represents the custom  dir of  the image saved
default_dir = 'bg_images'
actual_dir = default_dir
if len(sys.argv) > 2:
    actual_dir = sys.argv[2]
if not os.path.exists(actual_dir):
    os.mkdir(actual_dir)

class CountPeople:
    def __init__(self, pre_read_count=30, th_bgframes=20):
        # the counter of the bgframes
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.amg = adafruit_amg88xx.AMG88XX(self.i2c)
        self.grid_x, self.grid_y = np.mgrid[0:7:32j, 0:7:32j]
        self.bgframe_cnt = 0
        self.all_bgframes = []  # save all frames which sensor read
        self.pre_read_count = pre_read_count
        self.th_bgframes = th_bgframes
        self.image_id = 0  # the id of the hot image of each frame saved
        # 8*8 grid
        self.points = [(math.floor(ix/8), (ix % 8)) for ix in range(0, 64)]
        # discard the first and the second frame
        for i in range(self.pre_read_count):
            for row in self.amg.pixels:
                pass

    def interpolate(self, points, pixels, grid_x, grid_y, inter_type='cubic'):
        '''
        interpolating for the pixels,default method is cubic
        '''
        return griddata(points, pixels, (grid_x, grid_y), method=inter_type)

    def readPixelsArray(self):
        '''
          func:  read pixels
          args:none
          return :2-d array
        '''
        pass

    def displayImage_bg_curr(self, average_temperature, currFrameIntepol):
        '''
            show the background temperature and the frame of current
            temperature
        '''
        plt.subplot(1, 2, 1)
        plt.imshow(average_temperature)
        plt.title('background temperature')
        plt.subplot(122)
        print('subplot 122')
        plt.imshow(currFrameIntepol)
        plt.title('current temperature')
        fig, (bgaxes, cur_frames_axes,
              bg_cur_diff_axes) = plt.subplots(3, 1, num=2)
        print(bgaxes)
        bgaxes.set_xlim(16, 32)
        print('after xlim')
        bgaxes.hist(average_temperature.ravel(), bins=256, range=(
            17, 21), histtype='step', label='temperature hist')
        bgaxes.set_title('bg temperature list')
        print('hist')
        cur_frames_axes.hist(currFrameIntepol.ravel(), bins=256, range=(
            17, 28), histtype='step', label='current temperature')
        cur_frames_axes.set_title('curr temperature hist')
        diff = currFrameIntepol - average_temperature
        diff = np.round(diff, 1)
        print('cal diff')
        bg_cur_diff_axes.hist(diff.ravel(), bins=512, range=(
            -4, 4), histtype='step', label='difference between background temperature and current temperature')
        bg_cur_diff_axes.set_title(
            'difference between bg temperature and current temperature')
        fig.tight_layout()
        plt.show()

    def averageFilter(self, average_temp, curr_temp):
        '''
            if this frame is bg temperature
                return false
            else if this frame has human
                return False 
        '''
        bgAverage = np.round(np.average(average_temp), 1)
        currAverage = np.round(np.average(curr_temp), 1)
        diff = bgAverage - currAverage
        print('their difference is %.1f' % (diff))
        if diff <= .5:
            return False
        else:
            return True

    def calAverageTemp(self):
        '''
           func: calulate the temperature of n frames ,n >= 200
            args:none
            return : 2-d array which is the average temperature of
                    n frames
        '''
        if len(self.all_bgframes) < (self.th_bgframes):
            raise RuntimeError('the len of the all_bgframes is too small')
        total_frames = np.zeros((8, 8))
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
    def medianFilter(self,img):
        img =self.makeImgCompatibleForCv(img)
        median = cv.medianBlur(img,5)
        return median
    def gaussianFilter(self,img):
        img =self.makeImgCompatibleForCv(img)
        blur = cv.GaussianBlur(img,(5,5),0)
        return blur
    def bilateralFilter(self,img):
        img =self.makeImgCompatibleForCv(img)
        cv.bilateralFilter(img,9,75,75)
    def makeImgCompatibleForCv(self,img):
        cv.imwrite('temp.png',img)
        return cv.imread('temp.png',0)
    def otsuThreshold(self, img ):
        img =self.makeImgCompatibleForCv(img)
        ret,th = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return th
    def displayImage(self,img,title='temp'):
        #plt.ion()
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    def saveImage(self, average_temperature, currFrameIntepol):
        plt.subplot(1, 2, 1)
        plt.imshow(average_temperature)
        plt.title('background temperature')
        plt.subplot(122)
        print('subplot 122')
        plt.imshow(currFrameIntepol)
        plt.savefig("%s/hot_image_ %d.png" % (actual_dir ,self.image_id))
        self.image_id = self.image_id+1
        plt.clf()

    def process(self):
        '''
            main function
        '''

        # load the avetemp.py stores the average temperature
        # the result of the interpolating for the grid
        average_temperature = np.load("avgtemp.npy")
        all_frames = []
        try:
            while True:
                currFrame = []
                for row in self.amg.pixels:
                    # Pad to 1 decimal place
                    currFrame.append(row)
                currFrame = np.array(currFrame)
                all_frames.append(currFrame)
                print("current temperature is ")
                print(currFrame)
                currFrameIntepol = self.interpolate(
                    self.points, currFrame.flatten(), self.grid_x, self.grid_y, 'linear')
                # self.displayImage(average_temperature , currFrameIntepol)
                plt.figure(num=1)
                self.displayImage(currFrameIntepol,'original image')
                gblur =self.gaussianFilter(currFrameIntepol)
                plt.figure(num=2)
                self.displayImage(gblur,'gaussian filter image')
                median_filter = self.medianFilter(currFrameIntepol)
                plt.figure(num=3)
                self.displayImage(median_filter,'median filter image')
                th = self.otsuThreshold(gblur)
                plt.figure(num=4)
                self.displayImage(th,'otsu threshold')
                plt.show()
                break
               # self.averageFilter(average_temperature, currFrameIntepol)
        except KeyboardInterrupt:
            print("catch keyboard interrupt")
            print("length of the all_frames: %d"%(len(all_frames)))
            for i in range(len(all_frames)):
                currFrameIntepol = self.interpolate(
                    points, all_frames[i].flatten(), self.grid_x, self.grid_y, 'linear')
                self.saveImage(average_temperature, currFrameIntepol)
            print("save all frames")
            print("exit")

    def extractBody(self):
        pass

    def findBodyLocation(self):
        pass

    def trackPeople(self):
        pass


countp = CountPeople()
countp.process()
