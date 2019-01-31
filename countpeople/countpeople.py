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
from otsuBinarize import otsuThreshold

class ObjectTrack:
    '''
    目标轨迹类
    '''
    __loc_list = [] #运动轨迹队列
    __pos = -1
    __size = 0
    __direction = 1#进入
    def __init__(self):
        pass
    def put(self,point):
        self.__loc_list.append(point)
        self.__size += 1
    def get(self):
        loc = self.__loc_list[self.__size -1]
        return loc
    def isEmpty(self):
        return self.__size == 0 
    def hasPassDoor(self,frame_width = 32):
        '''
            是否已经通过
        '''
        xs,xend = self.__loc_list[0] ,self.__loc_list[self.__size-1]
        if xend[0] < xs[0] :
            self.__direction = 0 
        s = frame_width -1 
        dis = xend[0] - xs[0]
        return abs(dis)== s
    def getDirection(self):
        return self.__direction
class Target:
    '''
    运动目标
    '''
    __center_temperature = 0 #邻域平均温度
    def __init__(self ,center_temperature= 0 ):
        self.__center_temperature = center_temperature
    def getNeiborhoddTemp(self):
        return self.__center_temperature
    def setNeiborhoodTemp(self , neiborhood):
        self.__center_temperature = neiborhood
class CountPeople:
    __peoplenum = 0  # 统计的人的数量
    __diffThresh = 2# 温度差阈值
    __otsuThresh = 3.0 # otsu 阈值
    __averageDiffThresh = 0.3  # 平均温度查阈值
    __otsuResultForePropor = 0.0004
    __objectTrackDict = {}#目标运动轨迹字典，某个运动目标和它的轨迹映射
    __objectImgDict = {}#目标图片数据字典
    __neiborhoodTemperature = {}#m目标图片邻域均值
    __neibor_diff_thresh = 1
    # otsu阈值处理后前景所占的比例阈值，低于这个阈值我们认为当前帧是背景，否则是前景

    def __init__(self, pre_read_count=30, th_bgframes=200, row=32, col=32):
        # the counter of the bgframes
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.amg = adafruit_amg88xx.AMG88XX(self.i2c)
        self.grid_x, self.grid_y = np.mgrid[0:7:32j, 0:7:32j]
        self.bgframe_cnt = 0
        self.all_bgframes = []  # save all frames which sensor read
        self.pre_read_count = pre_read_count
        self.th_bgframes = th_bgframes
        self.row = row  # image's row
        self.col = col  # image's col
        self.image_size = row * col
        self.image_id = 0  # the id of the hot image of each frame saved
        self.hist_id = 0  # the id of the hist image of diff between average
        # temp and current temp
        # 8*8 grid
        self.points = [(math.floor(ix/8), (ix % 8)) for ix in range(0, 64)]
        self.otsu_threshold = 0
        print("size of image is (%d,%d)"%(self.row,self.col)) 
        print("imagesize of image is %d"%(self.image_size))
        #i discard the first and the second frame
    def preReadPixels(self,pre_read_count = 20):
        self.pre_read_count =  pre_read_count
        #预读取数据，让数据稳定
        for i in range(self.pre_read_count):
            for row in self.amg.pixels:
                pass
    def interpolate(self, points, pixels, grid_x, grid_y, inter_type='cubic'):
        '''
        interpolating for the pixels,default method is cubic
        '''
        return griddata(points, pixels, (grid_x, grid_y), method=inter_type)
    def displayImage_bg_curr(self, average_temperature, currFrameIntepol):
        '''
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
        bg_cur_diff_axes.hist(diff.ravel(), bins=512, range=(
            -4, 4), histtype='step', label='difference between background temperature and current temperature')
        bg_cur_diff_axes.set_title(
            'difference between bg temperature and current temperature')
        fig.tight_layout()
        plt.show()

    def calAverageAndCurrDiff(self, average_temp, curr_temp):
        '''
            calculate the difference between avereage temperature and current temperature
            args:None
            return : difference between average tempe and currtemp

        '''
        diff = np.array(curr_temp - average_temp,np.float32)
        return np.round(diff, 2)

    def isBgByAverageDiff(self, average_temp, curr_temp):
        '''
            if this frame is bg temperature
                return True
            else if this frame has human
                return False 
        '''
        bgAverage = np.round(np.average(average_temp), 1)
        currAverage = np.round(np.average(curr_temp), 1)
        diff = currAverage - bgAverage
        print('their difference is %.2f' % (diff))
        if diff <= .5:
            return True
        else:
            return False

    def calAverageTemp(self,all_bgframes):
        '''
           func: calulate the temperature of n frames ,n >= 200
            args:none
            return : 2-d array which is the average temperature of
                    n frames
        '''
        total_frames = np.zeros((8, 8))
        for aitem in range(len(self.all_bgframes)):
            total_frames = total_frames+np.array(all_bgframes[aitem])
        return total_frames/len(all_bgframes)

    def detectionNoise(self):
        '''
            func:remove noise from image
            args:none
            return :2-d array
        '''
        pass

    def medianFilter(self, img):
        img = self.makeImgCompatibleForCv(img)
        median = cv.medianBlur(img, 5)
        return median

    def gaussianFilter(self, img):
        img = self.makeImgCompatibleForCv(img)
        blur = cv.GaussianBlur(img, (3, 3), 0)
        return blur

    def bilateralFilter(self, img):
        img = self.makeImgCompatibleForCv(img)
        cv.bilateralFilter(img, 9, 75, 75)

    def makeImgCompatibleForCv(self, img,datatype=np.float32):
        img = np.array(img, datatype)
        return img
    def otsuThreshold(self, img):
        print("start otsu binarize")
        img = self.makeImgCompatibleForCv(img)
        mins = np.round(img.min())
        maxs = np.round(img.max())
        ret, th = otsuThreshold(img ,self.image_size ,ranges = (mins,maxs))
        if self.otsu_threshold:
            self.otsu_threshold = round(ret + self.otsu_threshold,1)
        else:
            self.otsu_threshold = round(ret , 1)
        self.otsu_th_mask = th
        return th

    def equalizeHist(self, img):
        '''
            Histogram equalization
        '''
        img = self.makeImgCompatibleForCv(img)
        return cv.equalizeHist(img)

    def displayImage(self, img, title='temp', gray=True):
        # plt.ion()
        if gray == True:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    def saveDiffHist(self, diff, histtype="step"):
        plt.subplot(2, 1, 1)
        plt.title("hist_%d.png" % (self.hist_id))
        print(diff.shape)
        plt.hist(diff.ravel(), bins=1200, range=(-6, 6), histtype=histtype)
        plt.ylabel("temperature(%)")
        plt.xlabel("temperature(oC)")

        plt.savefig("%s/diff_hist_%d.png" % (actual_dir, self.hist_id))
        self.hist_id += 1
        plt.tight_layout()
        plt.clf()

    def saveImage(self, average_temperature, currFrameIntepol, filter=False):
        num = (1, 2)
        if filter == True:
            num = (2, 3)
        row, col = num
        ax_id = 1
        plt.subplot(row, col, ax_id)
        ax_id += 1
        plt.imshow(average_temperature)
        plt.title('bgTemperature')
        plt.subplot(row, col, ax_id)
        ax_id += 1
        plt.imshow(currFrameIntepol)
        plt.title("curTemp original ")
        if filter == True:
            plt.subplot(row, col, ax_id)
            ax_id += 1
            gblur = self.gaussianFilter(currFrameIntepol)
            plt.imshow(gblur)
            plt.xticks([])
            plt.yticks([])
            plt.title('gaussian filter')
            plt.subplot(row, col, ax_id)
            ax_id += 1
            median = self.medianFilter(currFrameIntepol)
            plt.imshow(median)
            plt.xticks([])
            plt.yticks([])
            plt.title('median filter')
            plt.subplot(row, col, ax_id)
            ax_id += 1
            th = self.otsuThreshold(gblur)
            plt.imshow(th)
            plt.xticks([])
            plt.yticks([])
            plt.title('otsu thersholding')
        plt.tight_layout()
        plt.savefig("%s/hot_image_ %d.png" % (actual_dir, self.image_id))
        self.image_id = self.image_id+1
        plt.clf()
    def saveDiffBetweenAveAndCurr(self):
        counter = 0
        diff_queues.append(self.calAverageDiff(
            average_temperature, currFrameIntepol))
        if counter > 2000:
            diff_array = np.array(np.round(diff_queues, 1))
            average_diff = np.round(np.average(diff_array), 1)
            np.savetxt('diff_between_avgtemp_currtemp.txt',
                       diff_array, delimiter=',')
            np.savetxt('average_diff.txt', np.array([average_diff]))
            print('the average diff is %f' % (average_diff))
        counter += 1
        print('counter: %d' % (counter))

    def judgeFrameByHist(self, img):  # 根据直方图判断当前帧是否含有人类
        print("judge by hist")
        hist, bins = np.histogram(img.ravel(), bins=120, range=(-6, 6))
        print(len(hist))
        print(len(bins))
        bins = bins[:-1]
        diff = self.image_size - hist.sum()
        #将超过5.9的温度差的像素点的数目加到5.9中，这样让像素总数保持一致
        hist[-1] += diff
        freqMap = {}
        
        for i in range(bins.size):
            freqMap[bins[i]] = hist[i]
        sum = 0
        for i in bins:
            sum += freqMap[i]
        if sum > self.image_size / 10:
            return True
        else:
            return False

    # 根据当前温度和平均温度（表示背景温度)的差值判断是否含有人类
    def judgeFrameByDiffAndBTSU(self, img_diff):
        if img_diff.max() > self.__diffThresh:
            ret, hist = otsuThreshold(img_diff, 1024)
            fore_proportion = hist.sum() / self.image_size
            if fore_proportion > self.__otsuResultForePropor:
                return True
            return False
        else:
            return False

    # 根据当前温度的平均值和背景温度的平均值判断当前帧是否含有人类
    def judgeFrameByAverage(self, average_temp, current_temp):
        ave_ave = np.average(average_temp)
        curr_temp_ave = np.average(current_temp)
        diff_abs = abs(curr_temp_ave - ave_ave)
        if diff_abs > self.__averageDiffThresh:
            return True
        else:
            return False

    def isCurrentFrameContainHuman(self,current_temp,
            average_temperature,img_diff):
        '''
            判断当前帧是否含有人类
            ret : (bool,bool) ret[0] True:含有人类，ret[0] False:没有人类，表示属于背景
            ret[1] 为False丢弃这个帧，ret[1]为True，将这个帧作为背景帧
        '''
        hist_result  =  self.judgeFrameByHist(img_diff) 
        diff_result = self.judgeFrameByDiffAndBTSU(img_diff)
        ave_result = self.judgeFrameByAverage(average_temperature, current_temp)
        sums = [hist_result ,diff_result , ave_result]
        if sum(sums) == 3:
            return (True,)
        elif sum(sums) == 0:
            return (False,True)
        else:
            return (False,False)

    def acquireImageData(self,frame_count = 2000,customDir = None):
        '''
            这个方法是为了获得图像数据，数据保存在customDir中
        '''
        if customDir:
            if not os.path.exists(customDir):
                os.mkdir(customDir)
                print("create dir sucessfully: %s" % (customDir))
        else:
            customDir = "imagetemp"
            if not os.path.exists(customDir):
                os.mkdir(customDir)
        # load the avetemp.py stores the average temperature
        # the result of the interpolating for the grid
        average_path = customDir+"/"+"avgtemp.npy"
        print("the average path is %s" % (average_path))
        average_temperature = np.load(average_path)
        all_frames = []
        frame_counter = 0  # a counter of frames' num
        # diff_queues saves the difference between average_temp and curr_temp
        try:
            while True:
                currFrame = []
                for row in self.amg.pixels:
                    # Pad to 1 decimal place
                    currFrame.append(row)
                currFrame = np.array(currFrame)
                print("current temperature is ")
                print(currFrame)
                all_frames.append(currFrame)
                frame_counter += 1
                print("the %dth frame" % (frame_counter))
                if frame_counter > frame_count:
                    self.saveImageData(all_frames, customDir)
                    break

        except KeyboardInterrupt:
            print("catch keyboard interrupt")
            # save all images
            self.saveImageData(all_frames, customDir)
            print("save all frames")

    def process(self,  frame_interval=400):
        '''
            main function

            循环读取传感器数据，计算出背景温度后，读取的下一帧时，对帧进行插值处理，并进行相应的滤波处理（为了降噪），然后进一步
            噪声检测，通过计算图像直方图分布，计算背景温度差值并用大津二值法对差值图像进行二值化，下一步是计算帧的平均温度和背景温度的
            平均温度的差值，通过三者结合可以得出当前帧是否含有人类，如果存在人类那么可以进行下一步提取目标，找到目标位置，跟踪目标
            ，如果不存在就可以认为这个帧是背景温度，也可以丢弃这个帧(如果这个帧噪声很多，可能是其他人员路过监控区域边缘被检测到），
            这是为了减少噪声的影响，
            参数：frame_interval
            表示开始读取的数据帧数作为计算背景温度的数据，无人经过一段时间(1000帧之后)需要重新计算背景温度帧
        '''
        try:
            print("start running the application")
            self.calcBg = False #传感器是否计算完背景温度
            frame_counter = 0 #帧数计数器
            bg_frames = [] #保存用于计算背景温度的帧
            while True:
                currFrame = []
                for row in self.amg.pixels:
                    # Pad to 1 decimal place
                    currFrame.append(row)
                currFrame = np.array(currFrame)
                print("the %dth frame of the bgtemperature "%(len(bg_frames)+1))
                print("current temperature is ")
                print(currFrame)
                if frame_counter  ==  self.th_bgframes :
                    frame_counter = 0 
                    self.th_bgframes = 1000
                    #更新计算背景的阈值
                    self.average_temp = self.calAverageTemp(bg_frames)
                    #对平均温度进行插值
                    self.average_temp_intepol =  self.interpolate(self.points,self.average_temp.flatten(),self.grid_x,self.grid_y,'linear')
                    self.average_temp_median =   self.medianFilter(self.average_temp_intepol)
                    self.average_temp_gaussian =self.gaussianFilter(self.average_temp_intepol)
                    print("the new average temp's shape is "+str(self.average_temp_intepol.shape))
                    print("as list")
                    print(self.average_temp_intepol)
                    frame_counter = 0 # reset the counter
                    bg_frames = [] #清空保存的图片以节省内存
                    self.calcBg = True # has calculated the bg temperature

                elif not self.calcBg: #是否计算完背景温度
                    bg_frames.append(currFrame)
                    frame_counter += 1#帧数计数器自增
                    continue
                #计算完背景温度的步骤
                #对当前帧进行内插
                currFrameIntepol = self.interpolate(
                    self.points, currFrame.flatten(), self.grid_x, self.grid_y, 'linear')
                #对当前帧进行中值滤波，也可以进行高斯滤波进行降噪，考虑到分辨率太低，二者效果区别不大
                medianBlur = self.medianFilter(currFrameIntepol)
                #对滤波后的当前温度和平均温度进行差值计算
                temp_diff =self.calAverageAndCurrDiff(self.average_temp_median , medianBlur)
                ret =self.currentFrameContainHuman(medianBlur,self.average_temp_median ,temp_diff )
                if not ret[0]:
                    if ret[1]:
                        bg_frames.append(currFrame)
                        frame_counter += 1
                    continue
                #如果当前图片中含有两个人
                cnt_count,image ,contours,hierarchy =self.extractBody(self.average_temp_median , medianBlur)
                self
                print("当前帧数中存在的人数是%d"%(cnt_count))
                #下一步是计算轮当前帧的中心位置
                loc = self.findBodyLocation(medianBlur)
                print("location is on the  %dth row ")
                print(loc)
                #sleep(0.5)

        except KeyboardInterrupt:
            print("catch keyboard interrupt")
            # all_frames=[]
            # save all images
            self.saveImageData(all_frames, customDir)
            # for i in range(len(all_frames)):
            #print("shape is "+str(diff_queues[i].shape))
            #    self.saveDiffHist(diff_queues[i])
            #   self.saveImage(average_temperature ,all_frames[i],True)
            print("save all frames")
            print("exit")
            raise KeyboardInterrupt("catch keyboard interrupt")
   
    def extractBody(self,average_temp , curr_temp,show_frame = False):
        '''
           找到两人之间的间隙(如果有两人通过)
           在背景温度的基础上增加0.25摄氏度作为阈值,低于阈值作为背景，高于阈值作为前景，观察是否能区分两个轮廓，如果不能就继续循环增加0.25
           参数:average_temp:背景温度,curr_temp:被插值和滤波处理后的当前温度帧
        '''
        thre_temp = average_temp+0.25 #阈值温度
        ones = np.ones(average_temp.shape , np.float32)
        print("ones' shape is  (%d,%d)"%(average_temp.shape[0],average_temp.shape[1]))
        ret = (0 , None,None,None)
        area_1_3,area_1_10 = self.image_size*0.3,self.image_size*0.1
        while True:
            print("current threshold is ")
            print(thre_temp)
            print("current temperature is")
            print(curr_temp)
            binary = ones * (curr_temp >= thre_temp)
            bsum = binary.sum()
            proportion = round(bsum / self.image_size,2)
            print("binary_sum is %d"%(bsum))
            print("the propotion is %.2f"%(proportion))
            thresh = np.array(binary , np.uint8)
            print("after binarize :")
            print(thresh)
            img2 , contours , heirarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            print(contours)
            cont_cnt = len(contours)
            if cont_cnt == 0:
                return (0,None,None,None)
            print("has %d people"%(cont_cnt))
            #求轮廓的面积
            if cont_cnt  >= 1:
                cnt1 = cv.contourArea(contours[0])
                print("cnt1 = %.2f"%(cnt1))
            if len(contours) > 2:
                #暂时无法检测三个人以上进入
                thre_temp += 0.25
                continue
            if len(contours) > 1:
                #存在两个轮廓
                cnt2 = cv.contourArea(contours[1])
                print("cnt2= %.2f"%(cnt1))
                #如果两个轮廓的大小大于图片的1/10,那么可以认为存在两个人体
                img2_copy = img2.copy()
                #cv.drawContours(img2,contours,-1,(0,255,0),2)
                if cnt1 > area_1_10 and cnt2 > area_1_10:
                    print("return two people?!!!!!")
                    img2 = np.array(img2,np.float32)
                    if show_frame:
                        self.showExtractProcessImage(curr_temp,thresh ,img2)
                    return (2,img2_copy,contours,heirarchy)
            '''
            if cnt1 >  area_1_3:
                #如果区域面积大于图片1/3，则可能有两个人出现在图片上
                #增大阈值
                thre_temp += 0.25
            '''
            if cnt1 < area_1_10 and cont_cnt == 1 :
                img2_copy = img2.copy()
                #img_ret = cv.drawContours(img2,contours,-1,(0,255,0),1)
                print("has one people return !!!!!")
                img2 = np.array(img2,np.float32)
                rect = np.zeros(img2.shape,np.uint8)
                rect = cv.cvtColor(rect , cv.COLOR_GRAY2BGR)
                x,y,w,h = cv.boundingRect(contours[0])
                cv.rectangle(rect,(x,y),(x+w,y+h),(0,255,0),1)
                print(rect.sum())
                if show_frame:
                    self.showExtractProcessImage(curr_temp,thresh ,rect)
                return (1,img2_copy,contours,heirarchy)
            else:
                #不断提高阈值                    
                thre_temp += 0.25
        return ret
    def showExtractProcessImage(self,origin,thresh ,images_contours):
        #输出提取人体过程的图片
        print("=================the contours of the image==============")
        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(origin)
        ax1.set_title("origin")
        ax2.imshow(thresh)
        ax2.set_title("after thresh")
        ax3.imshow(images_contours)
        ax3.set_title("contours")
        plt.show()
    def saveImageData(self, all_frames, outputdir):
        print("length of the all_frames: %d" % (len(all_frames)))
        print("save all images data in "+outputdir+"/"+"imagedata.npy")
        # save all image data in directory:./actual_dir
        np.save(outputdir+"/imagedata.npy", np.array(all_frames))
        # save all diff between bgtemperature and current temperature in actual dir
    def findMaxLocation(self,img):
        row_max = []
        for i in range(len(img)):
            row_max.append(img[i].max())
        max_temp ,max_temp_row = row_max[0] ,0
        for i in range(1,len(row_max)):
            if row_max[i] >max_temp:
                max_temp = row_max[i]
                max_temp_row = i
        max_col_index= 0 
        for i in range(len(img[max_temp_row])):
            if max_temp == img[max_temp_row][i]:
                max_col_index = i
                break
        return (max_temp_row,max_col_index)

    def findBodyLocation(self,img,contours,xcorr ):
        '''
         找到人体位置，确定中心温度,通过计算每一行的最大温度可以确定人体的位置
         参数:img:当前帧 ， 
              contours:轮廓
        '''
        print("find body location")
        print("current temperature is ")
        #self.showContours(img,contours)
        print(np.round(img,2))
        pcount = len(contours)
        ret = []
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            print(x,y,w,h)
            mask = img[y:y+h,x:x+w]
            print(mask)
            row_sum = []
            for row in mask:
                row_sum.append(row.sum())
            max_row = row_sum.index(max(row_sum))
            mrow = mask[max_row].tolist()
            max_col = mrow.index(max(mrow))
            max_row += y
            max_col += x
            ret.append((max_row,max_col))
        print(ret)
        #input("press Enter continue...")
        return ret
        #for i in range(pcount):
            #cnt = contours[i]
            #moment = cv.moments(cnt)#求图像的矩
            #cx =int(moment['m10']/moment['m00'])
            #cy = int(moment['m01']/moment['m00'])
            #ret.append((cx,cy))
                #break
        '''
        imgs = img.copy()
        for i in range(pcount):
            res = self.findMaxLocation(imgs)
            imgs[res[0],res[1]] = 0
            ret.append(res)
        print("==================found the place of the center of temperature=================")
        print(ret)
        return ret
        img_after_otsu = img*self.otsu_th_mask
        print("otsu ")
        print(self.otsu_th_mask)
        print(img_after_otsu)
        for row in img_after_otsu:
            row_max.append(row.sum())
        #row_max_copy = row_max.copy()
        print(np.round(row_max,2))
        #找到最行的索引
        max_row_index =row_max.index(max(row_max))
        rowlist = img[max_row_index].tolist()
        max_col_index =rowlist.index(max(rowlist))
        print(max_col_index)
        #time.sleep(0.5)
        print("max_col_index's list is ")
        ret = [(max_row_index,max_col_index)]
        print("ret")
        print(ret)
        #time.sleep(2)
        #max_row 是最大值的下标
        if pcount > 1:
            row_max[max_row_index]= 0 
            max_row_index =[ row_max.index(max(row_max))]
            rowlist = img[max_row_index].tolist()
            max_col_index =rowlist.index(max(rowlist))
            ret.append((max_row_index,max_col_index))
        #self.paintHist(xcorr , row_max_copy, titles="row max hist",x_label = "row",y_label="row sum(。C)")
        return ret
        pass

        '''
    def showContours(self,img,contours):
        print("====================now paint the contours of the image========")
        img2 = np.array(img.copy(),np.uint8)
        cv.drawContours(img2,contours ,-1 ,(0,255,0),1)
        cv.imshow("contours",img2)
        cv.waitKey(0)
        time.sleep(10)
        cv.destroyAllWindows()
    def paintHist(self,xcorr,y,titles= "title",x_label="xlabel",y_label="ylabel",fig_nums = [100]):
        '''
        绘制直方图

        '''
        print("绘制直方图")
        print(xcorr)
        print(y)
        fig_nums[0] += 1
        fig,ax1 = plt.subplots(1,1,num=fig_nums[0])
        ax1.plot(xcorr,y)
        ax1.set_title(titles)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        plt.show()

    def setPackageDir(self, pdir):
        self.pdir = pdir
    def __neiborhoodTemp(self,img,points,nsize = 4):
        '''
        求邻域平均温度
        '''
        x,y = points
        count = 0 
        temp_sum = img[x][y]
        if nsize == 4:
            if x == 0:
                if y == 0 :
                    temp_sum += img[x+1][y]
                    temp_sum += img[x][y+1]
                    count=2
                elif y == 31:
                    temp_sum += img[x+1][y]
                    temp_sum += img[x][y-1]
                    count=2
                else:
                    temp_sum += img[x+1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x][y+1]
                    count=3
            elif x == 31:
                if y == 0 :
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y+1]
                    count=2
                elif y == 31:
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    count=2
                else:
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x][y+1]
                    count =3
            else:
                if y == 0 :
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y+1]
                    tmep_sum += img[x+1][y]
                    count=3
                elif y == 31:
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x+1][y]
                    count=3
                else:
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x][y+1]
                    temp_sum += img[x+1][y]
                    count=4
        elif nsize == 8:
            if x == 0:
                if y == 0 :
                    temp_sum += img[x+1][y]
                    temp_sum += img[x][y+1]
                    temp_sum +=img[x+1][y+1]
                    count=3
                elif y == 31:
                    temp_sum += img[x+1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x+1][y-1]
                    count=3
                else:
                    temp_sum += img[x+1][y-1]
                    temp_sum += img[x+1][y+1]
                    temp_sum += img[x+1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x][y+1]
                    count=5
            elif x == 31:
                if y == 0 :
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y+1]
                    temp_sum += img[x-1][y+1]
                    count=3
                elif y == 31:
                    temp_sum += img[x-1][y-1]
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    count=3
                else:
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x][y+1]
                    temp_sum += img[x-1][y-1]
                    temp_sum += img[x-1][y+1]
                    count =5
            else:
                if y == 0 :
                    temp_sum += img[x-1][y+1]
                    temp_sum += img[x+1][y+1]
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y+1]
                    tmep_sum += img[x+1][y]
                    count = 5
                elif y == 31:
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x+1][y]
                    temp_sum += img[x-1][y-1]
                    tmep_sum += img[x+1][y-1]
                    count = 5
                else:
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x][y+1]
                    temp_sum += img[x+1][y]
                    temp_sum += img[x-1][y+1]
                    temp_sum += img[x-1][y-1]
                    temp_sum += img[x+1][y-1]
                    temp_sum += img[x+1][y+1]
                    count = 8
            count+=1
        else:
            print("has only 2 choice :4 or 8")
        neibor = temp_sum / count
        return neibor
    def __extractFeature(self,img,corr):
        '''
        为当前帧提取目标特征
        '''
        #空间距离
        for cp in corr:
            for k,v in self.__objectTrackDict.items():
                last_place = v.get()
                euclidean_distance = math.sqrt(math.pow((cp[0] - last_place[0]),2)+math.pow((cp[1] - last_place[1]),2))
                print("euclidean distance is %.2f"%(euclidean_distance))
                #温度差不会超过1摄氏度
                previous_img = self.__objectImgDict[k]
                preivous_cnt = previous_img[last_place[0],last_place[1]]#前一帧的中心温度
                diff_temp = img[cp[0],cp[1]] - previous_cnt#当前中心温度和前一帧数的中心温度差
                #中心温度邻域均值
                ave = self.__neiborhoodTemp(img , cp[0],cp[1])
                heibor_diff = math.abs(ave - self._neiborhoodTemperature[k])
                if neibor_diff < self.__neibor_diff_thresh:
                    self.__objectTrackDict[k].put(cp)
                    self.__neiborhoodTempemperature[k] = (ave+self._neigborhoodTemperature[k])/2
                    self.__objectImgDict[k] = img
    def __classifyObject(self,img, corr):

        '''
        分类目标，确定当前点属于哪个目标

        '''
        if len(self.__objectTrackDict) == 0:
            track = ObjectTrack()
            for i in corr:
                obj = Target()
                track.put(i)
                self.__objectTrackDict[obj] = track
                ntp = self.__neigborhoodTemp(img,i[0],i[1])
                obj.setNeiborhoddTemp(ntp)
                self.__objectImgDict[k]=[img]
                self.__neiborhoodTemperature[k] =ntp
        else:
            self.__extractFeature(img,corr)
    def showTargetFeature(self):
        pass
    def __updateTrackStack(self,corr,classId):
        '''
        更新运动帧，更新目标的位置
        '''
    def __updatePeopleCount(self):
        '''
        更新检测的人数
        '''
        karr = []
        for k,v in self.__objectTrackDict.items():
            if v.hasPassDoor():
                if v.getDirection() > 0:
                    self.__peoplenum+=1
                else:
                    self.__peoplenum-=1
                karr.append(k)
        for k in karr:
            del self.__objectTractDict[k]
            del self.__objectImgDict[k]
            del self.__neiborhoodTemperature[k]
    def trackPeople(self,img , loc):
        '''
        loc:对象当前位置数组
        img:图片
        '''
        self.__classifyObject(img,loc)
        self.__updatePeopleCount()
        print("=========================people num is %d ==================="%(self.__peoplenum))

        pass
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            cp = CountPeople()
            cp.process()
    else:
        if len(sys.argv) > 2:
            if sys.argv[2] == 'people':
                print('test human,sleep 10 s')
                time.sleep(10)
        else:
            time.sleep(2)
        # sys.argv[2] represents the custom  dir of  the image saved
        current_dir = os.path.abspath(os.path.dirname(__file__))
        adjust_dir = current_dir
        if current_dir.endswith("grideye"):
            adjust_dir = current_dir + "/countpeople"
        packageDir = adjust_dir
        actual_dir = adjust_dir
        path_arg = ""
        if len(sys.argv) > 1:
            actual_dir = actual_dir + "/"+sys.argv[1]
            path_arg = sys.argv[1]+"/"
        # sleep 10 s
        print("the actual_dir is %s" % (actual_dir))

        if not os.path.exists(actual_dir):
            os.mkdir(actual_dir)
        countp = CountPeople()
        countp.preReadPixels()
        # 这是为了方便访问背景温度数据而保存了包countpeople的绝对路径
        countp.setPackageDir(packageDir)
        try:
            countp.acquireImageData()
        except KeyboardInterrupt("keyboard interrupt"):
            print("exit")
