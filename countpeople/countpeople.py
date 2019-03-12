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
    目标轨迹类,保存每个目标的在每一帧的位置
    '''
    def __init__(self):
        self.__loc_list = [] #运动轨迹队列
        self.__img_list = [] #运动所在的帧的数据，和上面的轨迹一一对应
        self.__pos = -1
        self.__size = 0
        self.__direction = 1#进入
        self.__compensation =2# 补偿值
    def put(self,point,img):
        self.__loc_list.append(point)
        self.__img_list.append(img)
        self.__size += 1
    def get(self):
        loc = self.__loc_list[self.__size -1]
        img = self.__img_list[self.__size-1]
        return loc,img
    def isEmpty(self):
        return self.__size == 0 
    def hasPassDoor(self,frame_width = 8):
        '''
            是否已经通过

        '''
        xs,xend = self.__loc_list[0] ,self.__loc_list[self.__size-1]
        if xend[1] < xs[1] :
            self.__direction = 0 
        s =frame_width -1 
        dis =abs( xend[1] - xs[1])+self.__compensation
        return dis >= s
    def getDirection(self):
        return self.__direction
    def showContent(self):
        print(self.__loc_list)
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
    def showContent(self):
        print(self.__center_temperature)
class CountPeople:
    # otsu阈值处理后前景所占的比例阈值，低于这个阈值我们认为当前帧是背景，否则是前景

    def __init__(self, pre_read_count=30, th_bgframes=100, row=8, col=8):
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
        self.diff_ave_otsu= 0.5#通过OTSU分类的背景和前景的差的阈值,用于判断是否有人
        print("size of image is (%d,%d)"%(self.row,self.col)) 
        print("imagesize of image is %d"%(self.image_size))
        #i discard the first and the second frame
        self.__diff_individual_tempera= 0.3 #单人进出时的中心温度阈值
        self.__peoplenum = 0  # 统计的人的数量
        self.__diffThresh = 2.5 #温度差阈值
        self.__otsuThresh = 3.0 # otsu 阈值
        self.__averageDiffThresh = 0.4 # 平均温度查阈值
        self.__otsuResultForePropor = 0.0004
        self.__objectTrackDict = {}#目标运动轨迹字典，某个运动目标和它的轨迹映射
        self.__neiborhoodTemperature = {}#m目标图片邻域均值
        self.__neibor_diff_thresh = 1
        self.__isExist = False #前一帧是否存在人，人员通过感应区域后的执行统计人数步骤的开关
        self.__image_area = (self.row-1)*(self.col-1)
        self.__hist_x_thresh = 2.0
        self.__hist_amp_thresh = 2
        self.__isSingle = False
        self.__var_thresh=0.125
        self.otsu_threshold =0
        self.interpolate_method='cubic'
    def preReadPixels(self,pre_read_count = 20):
        self.pre_read_count =  pre_read_count
        #预读取数据，让数据稳定
        for i in range(self.pre_read_count):
            for row in self.amg.pixels:
                pass
    def interpolate(self, points, pixels, grid_x, grid_y, inter_type=None):
        '''
        interpolating for the pixels,default method is cubic
        '''
        if not inter_type:
            inter_type = self.interpolate_method
        return griddata(points, pixels, (grid_x, grid_y), method=inter_type)
    def getPeopleNum(self):
        return self.__peoplenum
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
    def setSinglePeople(self,isSingle):
        self.__isSingle = isSingle
    def isSinglePeople(self):
        return self.__isSingle
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
        for i in range(len(all_bgframes)):
            total_frames = total_frames+np.array(all_bgframes[i])
        return total_frames/len(all_bgframes)
    def setExistPeople(self,exist=False):
        self.__isExist = exist
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
        #print("start otsu binarize")
        img = self.makeImgCompatibleForCv(img)
        mins = np.round(img.min())
        maxs = np.round(img.max())
        ret, binary= otsuThreshold(img ,self.image_size ,ranges = (mins,maxs))
        if self.otsu_threshold:
            self.otsu_threshold = round(ret + self.otsu_threshold,1)
        else:
            self.otsu_threshold = round(ret , 1)
        self.otsu_th_mask = ret
        return ret ,binary

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
            th ,binarize= self.otsuThreshold(gblur)
            plt.imshow(th)
            plt.xticks([])
            plt.yticks([])
            plt.title('otsu thersholding')
        plt.tight_layout()
        plt.savefig("%s/hot_image_ %d.png" % (actual_dir, self.image_id))
        self.image_id = self.image_id+1
        plt.clf()
    def judgeFrameByHist(self, img):  # 根据直方图判断当前帧是否含有人类
        print("judge by hist")
        hist, bins = np.histogram(img.ravel(), bins=120, range=(-6, 6))
        bins = bins[:-1]
        diff = self.image_size - hist.sum()
        #将超过5.9的温度差的像素点的数目加到5.9中，这样让像素总数保持一致
        hist[-1] += diff
        freqMap = {}
        for i in range(bins.size):
            freqMap[bins[i]] = hist[i]
        sums= 0
        for k,v in freqMap.items():
            if k > self.__hist_x_thresh:#直方图x轴阈值
                sums += v 
        print("=========================hist amp's sum is======================")
        print(sums)
        if sums >= self.__hist_amp_thresh:#直方图振幅阈值
            return True
        else:
            return False

    # 根据当前温度和平均温度（表示背景温度)的差值判断是否含有人类
    def judgeFrameByDiffAndBTSU(self, img_diff):
        ret, hist = self.otsuThreshold(img_diff.copy())
        print("=============otsu ret is %.2f ============="%(ret))
        hist = np.array(hist,np.uint8)
        print(hist)
        fg_img = img_diff[hist>0]
        bg_img = img_diff[hist==0]
        if len(bg_img) ==0 :
            return False
        fg_ave = np.average(fg_img)
        bg_ave = np.average(bg_img)
        diff_ave = fg_ave - bg_ave
        if diff_ave >= self.diff_ave_otsu:
            return True,hist
        else:
            return False,hist
    # 根据当前温度的平均值和背景温度的平均值判断当前帧是否含有人类
    def judgeFrameByAverage(self, average_temp, current_temp):
        ave_ave = np.average(average_temp)
        curr_temp_ave = np.average(current_temp)
        diff_abs = abs(curr_temp_ave - ave_ave)
        if diff_abs > self.__averageDiffThresh:
            return True
        else:
            return False
    def judgeFrameByDiffVar(self,diff_frame):
        var = np.var(diff_frame)
        if var >= self.__var_thresh:
            return True
        return False
    def isCurrentFrameContainHuman(self,current_temp,
            average_temperature,img_diff):
        '''
            判断当前帧是否含有人类
            ret : (bool,bool) ret[0] True:含有人类，ret[0] False:没有人类，表示属于背景
            ret[1] 为False丢弃这个帧，ret[1]为True，将这个帧作为背景帧
        '''
        #print(img_diff)
        var_result = self.judgeFrameByDiffVar(img_diff)
        hist_result  =  self.judgeFrameByHist(img_diff) 
        ave_result = self.judgeFrameByAverage(average_temperature, current_temp)
        sums = [hist_result ,var_result, ave_result]
        print(sums)
        if sum(sums) >=  2:
            return (True,)
        elif sum(sums) >0:
            return (False,False)
        else:
            print("case 3：=======no people=======")
            print(sums)
            return (False,True)

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

    def process(self,  frame_interval=400,testSubDir=None):
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
            time.sleep(2)
            self.preReadPixels()
            self.calcBg = False #传感器是否计算完背景温度
            frame_counter = 0 #帧数计数器
            seq_counter = 0 
            bg_frames = [] #保存用于计算背景温度的帧
            frame_with_human = []
            dir_counter = 1#目录计数器
            true_counter = 0#人出现的帧数
            all_frame=[]#所有帧数
            while True:
                currFrame = []
                for row in self.amg.pixels:
                    # Pad to 1 decimal place
                    currFrame.append(row)
                currFrame = np.array(currFrame)
                seq_counter += 1
                print("the %dth frame of the bgtemperature "%(seq_counter))
                print("current temperature is ")
                print(currFrame)
                if frame_counter  ==  self.th_bgframes :#是否测完平均温度
                    #更新计算背景的阈值
                    frame_counter=0
                    num = len(bg_frames)
                    print("====num is %d==="%(num))
                    self.average_temp = self.calAverageTemp(bg_frames)

                    ''' 
                    #对平均温度进行插值
                    self.average_temp_intepol =  self.interpolate(self.points,self.average_temp.flatten(),self.grid_x,self.grid_y,'cubic')
                    self.average_temp_median =   self.medianFilter(self.average_temp_intepol)
                    self.average_temp_gaussian =self.gaussianFilter(self.average_temp_intepol)
                    '''
                    bg_frames = [] #清空保存的图片以节省内存
                    self.calcBg = True # has calculated the bg temperature
                    print("===finish testing bg temperature===")
                    print("===average temp is ===")
                    print(self.average_temp)
                    continue

                elif not self.calcBg: #是否计算完背景温度
                    bg_frames.append(currFrame)
                    frame_counter += 1#帧数计数器自增
                    continue
                all_frame.append(currFrame)
                #计算完背景温度的步骤
                #对当前帧进行内插
                print("========================================================process============================================================")
                '''
                currFrameIntepol = self.interpolate(
                    self.points, currFrame.flatten(), self.grid_x, self.grid_y,'cubic')
                #对当前帧进行中值滤波，也可以进行高斯滤波进行降噪，考虑到分辨率太低，二者效果区别不大
                medianBlur = self.medianFilter(currFrameIntepol)
                #对滤波后的当前温度和平均温度进行差值计算
                temp_diff =self.calAverageAndCurrDiff(self.average_temp_median , medianBlur)
                '''
                diff_temp = self.calAverageAndCurrDiff(self.average_temp,currFrame)
                ret =self.isCurrentFrameContainHuman(currFrame,self.average_temp, diff_temp )
                if not ret[0]:
                    if self.getExistPeople():
                        '''
                        print("============restart calculate the bgtemp======")
                        self.calcBg=False
                        bg_frames = []#重置背景缓冲区
                        self.frame_counter =0 #重置背景帧数计数器
                        '''
                        self.setExistPeople(False)
                        self.__updatePeopleCount()
                    if ret[1]:
                        '''
                        bg_frames.append(currFrame)
                        frame_counter += 1
                        '''
                    else:#(False,False)
                        true_counter += 1
                    continue
                self.setExistPeople(True)
                (cnt_count,image ,contours,hierarchy),area =self.extractBody(self.average_temp, currFrame)
                if cnt_count ==0:
                    continue
                #下一步是计算轮当前帧的中心位置
                loc = self.findBodyLocation(diff_temp,contours,[ i for i in range(self.row)])
                self.trackPeople(diff_temp,loc)
                self.showPeopleNum() 
                #sleep(0.5)

        except KeyboardInterrupt:
            print("catch keyboard interrupt")
            print("true_counter =%d"%(true_counter))
            output_path = ""
            default_path = "test"
            if testSubDir:
                output_path += testSubDir
            else:
                output_path = default_path
            output_path +="/"
            if not  os.path.exists(output_path):
                os.mkdir(output_path)
            frame_output_path =output_path+ "imagedata.npy"
            avg_output_path = output_path +"avgtemp.npy"
            np.save(frame_output_path,np.array(all_frame))
            np.save(avg_output_path,self.average_temp)
            print("sucessfully save the image data")
            print("path is in "+output_path)
            # all_frames=[]
            # save all images
            #self.saveImageData(all_frames, customDir)
            # for i in range(len(all_frames)):
            #print("shape is "+str(diff_queues[i].shape))
            #    self.saveDiffHist(diff_queues[i])
            #   self.saveImage(average_temperature ,all_frames[i],True)
            #print("save all frames")
            print("exit")
            raise KeyboardInterrupt("catch keyboard interrupt")
    def showPeopleNum(self):
        print("=================current people num is %d ==============="%(self.__peoplenum))

    def getExistPeople(self):
        return self.__isExist
    def removeNoisePoint(self,curr_temp,corr):
        max_temperature_thresh=2
        horizontal_thresh = 2
        vertical_thresh = 2
        cp_temp_dict = {}
        corr_set = set(corr)
        corr_bak=corr_set.copy()
        for item in corr_set:
            if curr_temp[item] >=  max_temperature_thresh:
                cp_temp_dict[item] = curr_temp[item]
            else:
                corr_bak.remove(item)
        print("===============cp temp dict is======")
        print(cp_temp_dict)
        if len(corr_bak) <= 1:
            return list(corr_bak)
        cp_item_sorted =sorted(cp_temp_dict.items(),key =lambda d:d[1])
        cp_item_sorted=set(cp_item_sorted)
        reference_set =set()
        removed_set=set()
        print("====cp item sorted============")
        print(cp_item_sorted)
        for item in cp_item_sorted:
            reference_point=item
            if reference_point in reference_set or reference_point in removed_set:
                continue
            reference_set.add(reference_point)
            rest_points = cp_item_sorted - reference_set - removed_set
            for k in rest_points:
                hori_dis = abs(reference_point[0][1] -k[0][1])
                if hori_dis <=  horizontal_thresh :
                    vertical_dis = abs(reference_point[0][0] - k[0][0])
                    if vertical_dis <= vertical_thresh:
                        removed_set.add(k)
        final_corr = []
        rest_set = cp_item_sorted-removed_set
        print("================rest set is ======================")
        print(rest_set)
        for k,v in rest_set:
            final_corr.append(k)
        return final_corr
        #for item in corr:


    def extractBody(self , average_temp,curr_temp,show_frame=False,seq=None):
        thre_temp =average_temp.copy()+0.25
        ones = np.ones(average_temp.shape,np.float32)
        all_area =self.image_size
        while True:
            bin_img = ones*(curr_temp>= thre_temp)
            bin_img = bin_img.astype(np.uint8)
            #img , contours , heirarchy = cv.findContours(bin_img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            n , label = cv.connectedComponents(bin_img)
            area_arr = []
            label_dict= {}
            for i in range(1,n):
                sub_matrix = label[np.where(label==i)]
                area_arr.append(sub_matrix.size)
                label_dict[i] = sub_matrix.size
            area_arr.sort()
            if not area_arr:
                return (0,None,None,None),0
            max_area = area_arr[-1]
            if max_area >= all_area * 0.3:
                thre_temp += 0.25
            elif max_area  > all_area * 0.1:
                if len(area_arr) >=2 :
                    second_largest = area_arr[-2]
                    if second_largest > all_area*0.1:
                        label_sorted = sorted(label_dict.items() , key =lambda d:d[1])
                        sub_label = label_sorted[-2:]
                        key1,key2 = sub_label[0][0],sub_label[1][0]
                        if key1 != 1 and key2 != 1: 
                            label[np.where(label == 1)] = 0
                        if key1 != 1:
                            label[np.where(label ==key1)]=1
                        if key2 != 1:
                            label[np.where(label==key2)]=1
                        label[np.where(label!=1)] = 0
                        bin_img = label.astype(np.uint8)
                        img,contours,heir=cv.findContours(bin_img,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        con_area ={}
                        cont_area_arr = []
                        max_area = -1
                        for c in contours:
                            area = cv.contourArea(c)
                            while area  in con_area:
                                area += 0.1
                            cont_area_arr.append(area)
                            con_area[area] = c
                        ret_conto =[]
                        for a,c in con_area.items():
                            if a >= 1:
                                ret_conto.append(c)
                        return (2,img,ret_conto,heir),0
                    else:
                        thre_temp+=0.25
                else:
                    thre_temp += 0.25
            elif max_area  < math.ceil(all_area*0.1):
                label_sorted = sorted(label_dict.items(), key =lambda d:d[1])
                sub_label = label_sorted[-1]
                key = sub_label[0]
                if key != 1:
                    label[np.where(label ==1)] =0
                    label[np.where(label == key)] = 1
                label[np.where(label != 1)] = 0
                bin_img =label.astype(np.uint8)
                img,contours,heir=cv.findContours(bin_img,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                return (1,img,contours,heir),0
            else:
                thre_temp += 0.25


    def extractBodyBak(self,average_temp , curr_temp,show_frame = False,seq=None):
        '''
           找到两人之间的间隙(如果有两人通过)
           在背景温度的基础上增加0.25摄氏度作为阈值,低于阈值作为背景，高于阈值作为前景，观察是否能区分两个轮廓，如果不能就继续循环增加0.25
           参数:average_temp:背景温度,curr_temp:被插值和滤波处理后的当前温度帧
        '''
        #show_frame =True
        thre_temp = average_temp.copy()+1 #阈值温度
        ones = np.ones(average_temp.shape , np.float32)
        ret = (0 , None,None,None)
        area_down_thresh,thresh_up,thresh_down = self.__image_area*0.01,self.__image_area/9,self.__image_area/10
        kernel = np.ones((1,1))
        single_people_flag=False
        first_thresh = True
        while True:
            '''
            print("current threshold is ")
            print(thre_temp)
            print("current temperature is")
            print(curr_temp)
            '''
            #show_frame=True
            '''
            print("===curr_temp's sum is===")
            print(curr_temp.sum())
            print("===thre_temp's sum is ===")
            print(thre_temp.sum())
            print("===ones sum is===")
            print(ones.sum())
            '''
            #diff = curr_temp - thre_temp
            #diff[diff<0] = 0
            # 对图片进行ostu二值化
            ##th,bin_img = self.otsuThreshold(diff)
            '''
            print("after otsu binarize===")
            '''
            #plt.imshow(bin_img)
            #plt.show()
            bin_img = ones * (curr_temp >= thre_temp)
            '''
            bsum = binary.sum()
            proportion = round(bsum / self.image_size,2)
            print("binary_sum is %d"%(bsum))
            print("the propotion is %.2f"%(proportion))
            thresh = np.round(binary)
            thresh = np.array(thresh , np.uint8)
            #print("after binarize :")
            #print(thresh)
            '''
            bin_img = np.array(bin_img,np.uint8)
            #thresh = cv.erode(bin_img,kernel,iterations=1)#进行形态学侵蚀
            img2 , contours , heirarchy = cv.findContours(bin_img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            cont_cnt = len(contours)
            if show_frame:
                img2 = np.array(img2,np.float32)
                rect = np.zeros(img2.shape,np.uint8)
                rect = cv.cvtColor(rect , cv.COLOR_GRAY2BGR)
                x,y,w,h = cv.boundingRect(contours[0])
                cv.rectangle(rect,(x,y),(x+w,y+h),(0,255,0),1)
                self.showExtractProcessImage(curr_temp,img2 ,rect)
            if cont_cnt == 0:
                return (0,None,None,None),0
            #print("has %d people"%(cont_cnt))
            #求轮廓的面积
            area_dict = {}
            area_list = []
            for c in contours:
                area = cv.contourArea(c)
                while area  in area_dict:
                    area += 0.1
                area_dict[area] = c
                area_list.append(area)
            if first_thresh:
                first_thresh_sum = sum(area_list)
                print("=====first_thresh_sum=====")
                print(first_thresh_sum)
                if first_thresh_sum < 5:
                    first_thresh=False
                    single_people_flag=True
                    self.setSinglePeople(True)
            print("===========area list is =========")
            print(area_list)
            bool_arr_down = np.array(area_list) < thresh_down
            bool_arr_up = np.array(area_list) < thresh_up
            if all( bool_arr_down) or all(bool_arr_up) :
                print(area_list)
                ret = []
                for a in area_list:
                    if a > area_down_thresh:
                        ret.append(a)
                    else:
                        del area_dict[a]
                img2_copy = img2.copy()
                #cv.drawContours(img2,contours,-1,(0,255,0),2)
                cnts = []
                origin_num = len(ret)#原来的轮廓的数目
                erosion = cv.erode(bin_img,kernel,iterations=1)#进行侵蚀
                img2 , conts , heir = cv.findContours(erosion,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                erode_cnt = len(conts)#侵蚀后的轮廓数目
                if erode_cnt != origin_num and not single_people_flag:
                    erosion_area_list=[]
                    for c in conts:
                        erosion_area_list.append(cv.contourArea(c))
                    print("====ersion area list is====")
                    print(erosion_area_list)
                    print("===after erosing===")
                    return (erode_cnt,img2.copy,conts,heir),first_thresh_sum
                for k,v in area_dict.items():
                    cnts.append(v)
                pnum = len(ret)
                if pnum > 1:
                        print("=======only has one people======")
                        max_area_k = -1
                        for k,v in area_dict.items():
                            if k > max_area_k :
                                max_area_k = k
                            else:
                                try:
                                    id_arr = [id(i) for i in cnts]
                                    id_index = id_arr.index(id(v))
                                    cnts.pop(id_index)

                                except ValueError:
                                    print([id(item) for item in cnts])
                                    print(v)
                                    print(id(v))
                                    raise ValueError()

                        pnum = 1
                return (pnum,img2_copy,cnts,heirarchy),first_thresh_sum
            else:
                thre_temp += 0.25
                continue
            '''
            if cont_cnt  >= 1:
                cnt1 = cv.contourArea(contours[0])
                print("cnt1 = %.2f"%(cnt1))
            
            if len(contours) > 2:
                thre_temp += 0.25
                continue
        
            if cont_cnt  > 1:
                #存在两个轮廓以上
                print("cont_cnt > 1")
                area_arr = []
                for c in contours:
                    area_arr.append(cv.contourArea(c))
                if all(area_arr) > area_down_thresh and ( all(area_arr) < area_1_14 or all(area_arr) < area_1_15):
                    img2_copy = img2.copy()
                    #cv.drawContours(img2,contours,-1,(0,255,0),2)
                    img2 = np.array(img2,np.float32)
                    if show_frame:
                        self.showExtractProcessImage(curr_temp,thresh ,img2)
                    return (len(contours),img2_copy,contours,heirarchy)
            
            if cnt1 >  area_1_3:
                #如果区域面积大于图片1/3，则可能有两个人出现在图片上
                #增大阈值
                thre_temp += 0.25
            
            if cnt1 < area_1_14 :
                noMoreOne = False
                #这个条件是去除一些噪音所造成的误差
                if cont_cnt > 1:
                    area_arr = []
                    area_dict={cnt1:contours[0]}
                    for i in range(len(contours)):
                        area = cv.contourArea(contours[i])
                        while True:
                            if area  not in area_dict:
                                area_dict[area] = contours[i]
                                break
                            area += 0.1
                        area_arr.append(area)
                    area_arr.sort()
                    print("=====area arr is====")
                    print(area_arr)
                    max_area = area_arr[-1]
                    area_arr = area_arr[0:-1]
                    if all(area_arr) <= 3:
                        noMoreOne=True
                    contours=[area_dict[max_area]]
                    cnt1 = max_area

                if  cnt1 < area_1_14 and (cont_cnt ==1 or noMoreOne):
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
                    thre_temp+=0.25
            else:
                #不断提高阈值                    
                thre_temp += 0.25
        return ret
        '''
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
        #self.showContours(img,contours)
        #print(np.round(img,2))
        corr = []
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            mask = img[y:y+h,x:x+w]
            #input("press any key")
            row_max =[]
            for row in mask:
                row_max.append(row.max())
            row,col =0,0
            row = row_max.index(max(row_max))
            max_row = mask[row].tolist()
            col = max_row.index(max(max_row))
            row += y
            col += x
            temperature = img[row,col]
            max_temp = np.max(mask)
            print("====max_temp is ====")
            print(max_temp)
            if temperature != max_temp:
                xcorr,ycorr = np.where(mask == temperature)
                row = xcorr[0]+y
                col = ycorr[0]+x
            corr.append((row,col))
        print(corr)
        #print(img[ret[0][0],ret[0][1]])
        #input("press Enter continue...")
        print("================removed noise point===================")
        return self.removeNoisePoint(img,corr)
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
        #time.sleep(10)
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
                    temp_sum += img[x+1][y]
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
                    temp_sum += img[x+1][y]
                    count = 5
                elif y == 31:
                    temp_sum += img[x-1][y]
                    temp_sum += img[x][y-1]
                    temp_sum += img[x+1][y]
                    temp_sum += img[x-1][y-1]
                    temp_sum += img[x+1][y-1]
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
        print("=====all corr is=====")
        print(corr)
        obj_num = len(self.__objectTrackDict)#原来的目标数目
        updated_obj_set = set()#已经更新轨迹的目标集合 
        removed_point_set = set()#已经确认隶属的点的集合
        if len(self.__objectTrackDict) == 1:
           if len(corr) == 1:
               for k, v in self.__objectTrackDict.items():
                   last_place,last_frame = v.get()
                   last_tempera = last_frame[last_place[0],last_place[1]]
                   curr_tempera = img[corr[0][0],corr[0][1]]
                   diff = curr_tempera - last_tempera
                   if diff <=self.__diff_individual_tempera:
                       v.put(corr[0],img)
                       return 
        for cp in corr:
            for k,v in self.__objectTrackDict.items():
                last_place ,last_frame = v.get()
                #euclidean_distance = math.sqrt(math.pow((cp[0] - last_place[0]),2)+math.pow((cp[1] - last_place[1]),2))
                #print("euclidean distance is %.2f"%(euclidean_distance))
                #温度差不会超过某个阈值
                previous_img = last_frame
                previous_cnt = previous_img[last_place[0],last_place[1]]#前一帧的中心温度
                diff_temp = abs(img[cp[0],cp[1]] - previous_cnt)#当前中心温度和前一帧数的中心温度差
                #中心温度邻域均值
                #ave = self.__neiborhoodTemp(img , cp)
                #heibor_diff = abs(ave - self.__neiborhoodTemperature[k])
                horizontal_dis =abs(cp[1] - last_place[1])
                vertical_dis = abs(cp[1] - last_place[1])
                if vertical_dis < self.row * 5/ 12  and horizontal_dis < self.col/3 :
                    if  k not in updated_obj_set and cp not in removed_point_set:#防止重复更新某些目标的点
                        if not self.belongToEdge(cp) and not self.belongToEdge(last_place):
                            if diff_temp > 1.2 and not self.isSinglePeople:
                                continue#非边缘目标的中心的点温度之间的差距不能大于1.2
                        self.__objectTrackDict[k].put(cp,img)
                        #self.__neiborhoodTemperature[k] = (ave+self.__neiborhoodTemperature[k])/2
                        updated_obj_set.add(k)
                        removed_point_set.add(cp)
        obj_length = len(updated_obj_set)
        obj_set = set(self.__objectTrackDict.keys())
        obj_rest = obj_set - updated_obj_set#剩余的未被更新轨迹的对象
        point_rest = set(corr)-removed_point_set#剩余的点
        print("====obj rest is ====")
        print(obj_rest)
        print("====point rest is ===")
        print(point_rest)
        final_point_rest= point_rest.copy()#最终剩余的点，表示新进入的目标，位于视野边缘
        final_obj_rest = obj_rest.copy()#最终剩余的目标，表示该目标消失，通过了监控区域
        print(obj_rest)
        print(final_obj_rest)
        if obj_length < obj_num:#是否还有目标尚未匹配
            for point in point_rest :
                for obj in obj_rest :
                    if obj not in final_obj_rest or point not in final_point_rest :#如果目标已经更新了轨迹
                        print("==================not in final_obj_rest=================")
                        continue
                    v = self.__objectTrackDict[obj]
                    prev_point , prev_img = v.get()
                    diff_temp = abs(prev_img[prev_point[0],prev_point[1]] - img[point[0],point[1]])
                    hozi_dis = abs(prev_point[1]-point[1])
                    verti_dis = abs(prev_point[0]-point[0])
                    if hozi_dis < (self.col *5/12) and verti_dis < self.row *5/12:
                        if not self.belongToEdge(prev_point) and not self.belongToEdge(point):#中心点不在边缘
                            if diff_temp > 1.6:
                                print("===diff_temp > 1.5 ====",diff_temp)
                                continue#放松限制条件（由于传感器误差和计算误差）
                        self.__objectTrackDict[obj].put(point,img)
                        final_point_rest.remove(point)
                        final_obj_rest.remove(obj)
                    else:
                        print("=============================second match fail==============================")
        print("===final point_rest====")
        print(final_point_rest)
        if len(final_point_rest)> 0:#是否有新的人进入监控视野
            for point in final_point_rest:
                if self.nearlyCloseToEdge(point):
                    obj = Target()
                    v = ObjectTrack()
                    v.put(point,img)
                    print("new one entering the visual field")
                    v.showContent()
                    self.__objectTrackDict[obj]= v
        if len(final_obj_rest) > 0 :#证明有些人已经通过监控区域
            self.updateSpecifiedTarget(final_obj_rest)
    def nearlyCloseToEdge(self,point):#近似作为边界
        if point[1]+5 >= self.row-1 or point[1]-5 <= 0:
            return True
        return False

    def updateSpecifiedTarget(self,key):#某个目标突然消失，表示通过监控区域
        for k in key:
            track = self.__objectTrackDict[k]
            if track.hasPassDoor(self.col):
                if track.getDirection() == 1:
                    self.__peoplenum += 1
                else:
                    self.__peoplenum -= 1
                del self.__objectTrackDict[k]
            else:
                print("no pass door")
                del self.__objectTrackDict[k]

    def belongToEdge(self,point):
        if point[1] == 0 or point[1] == self.col-1:
            return True
        return False
    def showFeature(self):
        pass
    def __classifyObject(self,img, corr):

        '''
        分类目标，确定当前点属于哪个目标

        '''
        print("======extract feature=====")
        self.__extractFeature(img,corr)
        self.showTargetFeature()

    def showTargetFeature(self):
        print("====show target feature===")
        num = len(self.__objectTrackDict)

        for k,v in self.__objectTrackDict.items():
            print(k,end=",")
            v.showContent()
    def __updateTrackStack(self,corr,classId):
        '''
        更新运动帧，更新目标的位置
        '''
    def updatePeopleCount(self):
        self.__updatePeopleCount()
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
            del self.__objectTrackDict[k]
            #del self.__objectImgDict[k]
            #del self.__neiborhoodTemperature[k]
    def trackPeople(self,img , loc):
        '''
        loc:对象当前位置数组
        img:图片
        '''
        self.__classifyObject(img,loc)
        #print("=========================people num is %d ==================="%(self.__peoplenum))
if __name__ == "__main__":
    if len(sys.argv) > 1 :
        if sys.argv[1] == "start":
            cp = CountPeople()
            interval = 400
            outputSubDir=None
            if len(sys.argv) > 2:
                outputSubDir =  sys.argv[2]
            cp.process(interval , outputSubDir)
        elif sys.argc[1] == "collect":
            if len(sys.argv)>2:
                subdir =""
                if sys.argv[2] == "delay":
                    print('delay 10 s')
                    time.sleep(10)
                    # sys.argv[2] represents the custom  dir of  the image saved    
                    if len(sys.argv) > 3:
                        subdir = sys.argv[2]
                else:
                    subdir = sys.argv[1]
                current_dir = os.path.abspath(os.path.dirname(__file__))
                adjust_dir = current_dir
                if current_dir.endswith("grideye"):
                    adjust_dir = current_dir + "/countpeople"
                packageDir = adjust_dir
                actual_dir = adjust_dir
                path_arg = ""
                if len(sys.argv) > 3:
                    actual_dir = actual_dir + "/"+subdir
                    path_arg = subdir+"/"
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
