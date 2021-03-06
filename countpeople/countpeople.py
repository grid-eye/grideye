import numpy as np
import cv2 as cv
import time
import math
import scipy
import os
import sys
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from otsuBinarize import otsuThreshold
from knn import createTrainingSetAndTestSet,knnClassify,getDefaultBgpathAndFgpath
from objecttrack import ObjectTrack
from target import Target
try:
    import busio
    import adafruit_amg88xx
    import board
    load_busio = True
except (ImportError,NotImplementedError):
    print("no busio or board")
    load_busio = False
class CountPeople:
    # otsu阈值处理后前景所占的比例阈值，低于这个阈值我们认为当前帧是背景，否则是前景

    def __init__(self, pre_read_count=30, th_bgframes=128, row=8, col=8,load_amg = False):
        # the counter of the bgframes
        if __name__ == "__main__" and load_busio or load_amg:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.amg = adafruit_amg88xx.AMG88XX(self.i2c)
        self.grid_x, self.grid_y = np.mgrid[0:7:32j, 0:7:32j]
        self.all_bgframes = []  # save all frames which sensor read
        self.pre_read_count = pre_read_count
        self.th_bgframes = th_bgframes
        self.row = row  # image's row
        self.col = col  # image's col
        self.image_size = self.row*self.col#固定为amg8833的数据大小
        self.image_id = 0  # the id of the hot image of each frame saved
        self.hist_id = 0  # the id of the hist image of diff between average
        # temp and current temp
        # 8*8 grid
        self.points = [(math.floor(ix/8), (ix % 8)) for ix in range(0, 64)]
        self.diff_ave_otsu= 0.5#通过OTSU分类的背景和前景的差的阈值,用于判断是否有人
        self.calcBg = False #传感器是否计算完背景温度
        print("size of image is (%d,%d)"%(self.row,self.col)) 
        print("imagesize of image is %d"%(self.image_size))
        #i discard the first and the second frame
        self.__x_thresh =2#用于提取特征的阈值
        self.__y_thresh =2
        self.__peoplenum = 0  # 统计的人的数量
        self.__enter = 0
        self.__exit = 0
        self.__entrance_exit_events = 0
        self.__diffThresh = 2.5 #温度差阈值
        self.__otsuThresh = 3.0 # otsu 阈值
        self.__averageDiffThresh = 0.4 # 平均温度查阈值
        self.__otsuResultForePropor = 0.0004
        self.__objectTrackDict = {}#目标运动轨迹字典，某个运动目标和它的轨迹映射
        self.__isExist = False #前一帧是否存在人，人员通过感应区域后的执行统计人数步骤的开关
        self.__entry_exit_events=0
        self.__hist_x_thresh = 2.0
        self.__hist_amp_thresh = 2
        self.__isSingle = False
        self.__var_thresh=0.125
        self.__k = 7
        self.image_thresh = 16#提取目标轮廓用的阈值
        self.stop_extract_h = 4#查找轮廓停止迭代的轮廓高度值
        self.single_people_area = 8#单人轮廓面积阈值,如果轮廓大小不超过这个值我们认为这是单个目标
        self.save_contours_row = 8#是否满足保存轮廓的条件
       # self.single_dog_max_y = 5
        self.hozi_double_check_thresh = 4
        self.verti_double_check_thresh = 4
        self.__street_dis_thresh = 5
        self.M = 50#计算背景帧的数量,用于构造均值背景模型
        self.otsu_threshold =1
        self.isDoorHigh=True
        self.interpolate_method='cubic'#内插方法
        self.bg_path,self.fg_path = getDefaultBgpathAndFgpath() 
        self.createTrainSample(self.bg_path,self.fg_path)
        self.sampleNum = 20#vibe背景模型大小
        self.bgUpdateProbability = 16#vibe背景模型更新概率
        self.minMatchBg = 4#vibe背景模型，不超过R的次数T，决定当前像素是前景还是背景
        self.bgRadius = 0.3 #vibe背景模型，当前像素和前景像素之间的温度差阈值R
        self.continueBgThresh = 50
        #self.initVibeModel()
    def setRow(self,row):
        self.row = row
        self.image_size = self.row * self.col
    def setCol(self,col):
        self.col = col
        self.image_size = self.row * self.col
    def preReadPixels(self,pre_read_count = 20):
        self.pre_read_count =  pre_read_count
        #预读取数据，让数据稳定
        for i in range(self.pre_read_count):
            for row in self.amg.pixels:
                pass
    def isCalcBg(self):
        return self.calcBg
    def setCalcBg(self,calc):
        self.calcBg = calc
    def getBgTemperature(self):
        return self.average_temp
    def showBgModel(self):
        for i in range(self.bgModel.shape[0]):
            for j in range(self.bgModel.shape[1]):
                print(self.bgModel[i][j])
    def initVibeModel(self):#初始化vibe前景和vibe背景模型
        self.bgModel = np.zeros((self.row,self.col,self.sampleNum+1))#加1是为了保存该像素点被认为是前景的次数
        self.fgModel = np.zeros((self.row,self.col),np.uint8)
        self.neigborCorr = [-1,0,-1,0,-1,0,1,1,1]
    def constructVibeBgModel(self,bgImg):#通过背景帧构建背景模型
        self.average_temp = bgImg
        for i in range(bgImg.shape[0]):
            for j in range(bgImg.shape[1]):
                for s in range(0,self.sampleNum):
                    x = random.choice(self.neigborCorr)
                    y = random.choice(self.neigborCorr)
                    row = i + x
                    col = j+y
                    if row < 0 :
                        row = 0 
                    elif row > self.row-1:
                        row = self.row -1 
                    if col < 0 :
                        col  = 0 
                    elif col  > self.col -1 :
                        col = self.col -1 
                    self.bgModel[i][j][s] = bgImg[row][col]
    def getBgModel(self):
        return self.bgModel
    def getFgModel(self):
        return self.fgModel
    def updateVibeBgModel(self,img):
        return 
        print("update bg model")
        sel_list = [i for i in range(0,self.sampleNum)]
        for i in range(self.bgModel.shape[0]):
            for j in range(self.bgModel.shape[1]):
                matches  = 0 
                dis = abs ( img[i][j] - self.bgModel[i][j][:-1] )
                for d in dis:
                    if d <  self.bgRadius:
                        matches += 1
                    if matches == self.minMatchBg:
                        break
                if matches >= self.minMatchBg:
                    self.bgModel[i][j][self.sampleNum] = 0
                    #如果像素是背景像素，那么有1/self.bgUpdateProbability的概率更新自己的模型样本值
                    updateBg = random.randint(0,self.bgUpdateProbability-1 ) == 0
                    if updateBg :
                        v = random.choice(sel_list)
                        #以当前像素随机更新背景样本库中20个样本中任意一个值
                        if img[i][j] == 0:
                            img[i][j] = self.average_temp[i][j]
                        self.bgModel[i][j][v] = img[i][j]
                    #同时以同样的概率更新它的邻居点模型样本的值
                    updateNeigbor = random.randint(0,self.bgUpdateProbability-1) == 1
                    if updateNeigbor:
                        #随机更新(i,j)的邻居点的背景样本库
                        row = i + random.choice(self.neigborCorr)
                        col = j + random.choice(self.neigborCorr)
                        if row < 0:
                            row = 0
                        elif row > self.row -1 :
                            row = self.row -1 
                        if col < 0 :
                            col = 0
                        elif col > self.col-1 :
                            col = self.col-1 
                        v = random.choice(sel_list)
                        if img[i][j] == 0:
                            img[i][j] = self.average_temp[i][j]
                        self.bgModel[row][col][v] = img[i][j]
                    self.fgModel[i][j] = 0
                else:
                    #当前像素是前景
                    self.bgModel[i][j][self.sampleNum] += 1
                    self.fgModel[i][j] = 1
                    if self.bgModel[i][j][self.sampleNum] > self.continueBgThresh:#一个像素连续50次被连续检测为前景，则认为该静止区域为运动区域，将其更新为背景点
                        self.bgModel[i][j][self.sampleNum] = 0
                        self.fgModel[i][j] = 0
                        rand = random.choice(sel_list)#随机选择背景模型集的任意一个元素数值
                        if img[i][j] == 0:
                            raise ValueError()
                        if img[i][j] == 0:
                            img[i][j] = self.average_temp[i][j]
                        self.bgModel[i][j][rand] = img[i][j]
        temp = self.average_temp
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if self.bgModel[i][j][self.sampleNum] == 0:
                    isUpdate = random.randint(0,self.bgUpdateProbability-1 ) == 1
                    if isUpdate:
                        rand = random.randint(0,self.sampleNum-1)
                        temp[i][j] =self.bgModel[i][j][rand]
                        if temp[i][j] < 1:
                            raise ValueError()
        print("exit update bg model")
        if np.any(temp[i][j] <= 0):
            raise ValueError()
    def getVibeFgModel(self):
        return self.fgModel
    def getVibeBgModel(self):
        return self.bgModel

    def preReadBgTemperature(self,bg_number=400,output_dir ="temp"):
        output_dir = self.ensurePathValid(output_dir)
        counter = 0 
        all_frame = []
        while counter < bg_number:
            curr_frame = []
            for row in self.amg.pixels:
                curr_frame.append(row)
            all_Frame.append(cframe)
            counter += 1
        self.average_temp = self.calAverageTemp(all_frame)
        np.save(output_dir+"/avgtemp.npy",self.average_temp)
    def createTrainSample(self,bg_path,fg_path):
        bg_sample,fg_sample =  createTrainingSetAndTestSet(bg_path,fg_path)
        self.knnSampleSet = np.append(bg_sample,fg_sample,axis= 0 )
        return self.knnSampleSet
    def interpolate(self, points, pixels, grid_x, grid_y, inter_type=None):
        '''
        interpolating for the pixels,default method is cubic
        '''
        if not inter_type:
            inter_type = self.interpolate_method
        return griddata(points, pixels, (grid_x, grid_y), method=inter_type)
    def getEnterNum(self):
        return self.__enter
    def getExitNum(self):
        return self.__exit
    def getPeopleNum(self):
        return self.__peoplenum
    def getEntranceExitEvents(self):
        return self.__entrance_exit_events
    def setSinglePeople(self,isSingle):
        self.__isSingle = isSingle
    def isSinglePeople(self):
        return self.__isSingle
    def calAverageTemp(self,all_bgframes):
        '''
           func: calulate the temperature of n frames ,n >= 200
            args:none
            return : 2-d array which is the average temperature of
                    n frames
        '''
        all_bgframes = np.array(all_bgframes)
        res = np.round(np.average(all_bgframes,axis=0),2)
        return res
    def setExistPeople(self,exist=False):
        self.__isExist = exist
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
    def calculateImgFeature(self,diff_frame):
        var =np.var(np.ravel(diff_frame.ravel()))
        return (None,var , None)
    def vibeJudge(self):#本项目不适用此方法，因为像素分辨率太低，选择的邻域相互影响太大。
        cnt = np.sum(self.fgModel)
        if cnt < 2 :
            print("no false")
            return False
        else:
            return True
    def knnJudgeFrameContainHuman(self,current_temp,avgtemp,diff_frame,showVoteCount=False):
        feature_vector = self.calculateImgFeature(diff_frame)
        print(self.knnSampleSet.shape)
        trainSet =self.knnSampleSet[:,1]
        trainLabels = self.knnSampleSet[:,2]
        category , voteCount= knnClassify(trainSet,trainLabels,feature_vector,(1,),self.__k)
        if showVoteCount:
            print(category)
        if category == 1:
            return True
        else:
            return False
    def constructGaussianBgModel(self,first_frame):
        row,col =  first_frame.shape
        self.alpha_gaussian = 0.03#学习率
        stdInit = 9
        varInit = stdInit ** 2
        self.lambda_gaussian = 2.5*1.2#背景更新参数
        self.u_gaussian =first_frame.copy()
        self.average_temp = self.u_gaussian
        self.d_gaussian =np.zeros((row,col))
        self.std_gaussian = np.zeros((row,col))
        self.std_gaussian.fill(stdInit)
        self.var_gaussian =  np.zeros((row,col))
        self.var_gaussian.fill(varInit)
        return self.u_gaussian
    def updateGaussianBgModel(self,frame):
        for i in range(self.row):
            for j in range(self.col):
                gray  = frame[i][j]
                diff_abs =abs( gray - self.u_gaussian[i][j] )
                if diff_abs < self.lambda_gaussian*self.std_gaussian[i][j]:
                    self.u_gaussian[i][j] = (1-self.alpha_gaussian)*self.u_gaussian[i][j]+self.alpha_gaussian * frame[i][j]
                    self.var_gaussian[i][j] = (1-self.alpha_gaussian)*self.var_gaussian[i][j] + self.alpha_gaussian*(frame[i][j] - self.u_gaussian[i][j])**2
                    self.std_gaussian[i][j]  = self.var_gaussian[i][j]**0.5
                else:
                    self.d_gaussian[i][j] = frame[i][j] - self.u_gaussian[i][j]
    '''
    def constructMultiGaussianBgModel(self,current_frame):
        pixel_range = 80
        self.C_mgaussian = 3
        self.M_mgaussian = 3
        self.D_mgaussian = 2.5
        self.alpha_mgaussian = 0.01#学习率
        self.thresh_mgaussian = 0.25 #前景阈值
        self.sd_init_mgaussian = 15
        self.average_temp = current_frame#第一帧作为背景帧
        self.fg_mgaussian = np.zeros((self.row,self.col))
        self.bg_mgaussian = np.zeros((self.row,self.col))
        self.weight_mgaussian = np.zeros((self.row,self.col,self.C_mgaussian))#权重矩阵
        self.mean_mgaussian = np.zeros((self.row,self.col,self.C_mgaussian))#像素均值
        self.u_diff_mgaussian = np.zeros((self.row,self.col,self.C_mgaussian))
        self.sd_mgaussian = np.zeros((self.row,self.col,self.C_mgaussian))
        self.p_mgaussian = self.alpha_mgaussian / (1/self.C_mgaussian)#初始化P变量
        self.rank_gauss = np.zeros((1,self.C_mgaussian))#各个高斯分布的权重优先级
        for i in range(self.row):
            for j in range(self.col):
                for k in range(self.C_mgaussian):
                    self.mean_mgaussian[i][j][k] = np.random.random()*pixel_range
                    self.weight_mgaussian[i][j][k] = 1/self.C_mgaussian
                    self.sd_mgaussian[i][j][k] = self.sd_init_mgaussian
    def updateMgaussianBgModel(self,current_frame):
        for m in range(self.C_gaussian):
            self.u_diff_mgaussian[m] = abs(cruuent_frame - self.mean_mgaussian[:,:,m])

        #更新高斯模型的参数
        for i in range(self.row):
            for j in range(self.col):
                match = 0 
                for k in range(self.C_mgaussian):
                    if abs(self.u_diff_mgaussian[i][j][k]) <= self.D_mgaussian*self.sd_mgaussian[i][j][k]:
                        #更新权重，均值，标准差，p
                        match = 1
                        self.weight_mgaussian[i][j][k] = (1-self.alpha_mgaussian)*self.weight_mgaussian[i][j][k] + self.alpha_mgaussian
                        self.p_mgaussian = self.alpha_mgaussian/self.weight_mgaussian[i][j][k]
                        self.mean_mgaussian[i][j][k] = (1-self.p_mgaussian)*self.mean_mgaussian[i][j][k] + self.p_mgaussian * current_frame[i][j]
                        self.sd_mgaussian[i][j][k] = math.sqrt((1-self.p_mgaussian)*(self.sd_mgaussian[i][j][k]**2) + self.p_mgaussian*(current_frame[i][j] - self.mean_mgaussian[i][j][k])**2)
                    else:
                        self.weight_mgaussian[i][j][k] = (1-self.alpha_mgaussian)*self.weight_mgaussian[i][j][k]
                self.average_temp[i][j] = 0 
                for k  in range(self.C_mgaussian):
                    self.average_temp[i][j] = self.average_temp[i][j] + self.mean_mgaussian[i][j][k] * self.weight_mgaussian[i][j][k]
                if match == 0:
                    min_val = np.min(self.weight_gaussian[i][j])
                    min_index =np.where(self.weight_gaussian[i][j] == min_val)[0][0]
                    self.mean_mgaussian[i][min_index] = current_frame[i][j]
                    self.sd_mgaussian[i][j][min_index] = self.sd_init_mgaussian 
                self.rank_mgaussian = self.weight_mgaussian[i][j]
    '''
    def constructAverageBgModel(self, bg_frames):#构造平均背景模型
        ft = np.zeros((self.row,self.col))
        self.step = 3
        self.M = len(bg_frames)
        u_diff = ft.copy()
        diff_std = ft.copy()
        ft_arr = []
        for s in range(self.step,self.M,self.step):
            ft =abs( bg_frames[s] - bg_frames[s-self.step] )
            u_diff += ft
            ft_arr.append(ft)
        u_diff = u_diff / self.M#均值
        for f in ft_arr:
            diff_std += (f - u_diff)**2
        diff_std /= self.M
        diff_std = diff_std ** (1/2)
        self.u_diff = u_diff #差值均值
        self.u = self.calAverageTemp(bg_frames)
        self.average_temp = self.u
        self.diff_std = diff_std#标准差
        self.beta = 2
        self.TH = u_diff + self.beta * diff_std
        self.alpha = 0.5 #背景模型学习率
        return self.u
    def updateAverageBgModel(self,current_frame,last_frame_step):#更新背景模型
        bgModel = np.zeros(current_frame.shape)
        diff = abs(current_frame - self.average_temp)
        bgModel[np.where(diff  > self.TH)] = 1
        F = abs(current_frame - last_frame_step)
        u_temp =  (1-self.alpha)*self.average_temp + self.alpha * current_frame
        u_diff_temp =  (1-self.alpha)*self.u_diff + self.alpha*F
        diff_std_temp =  (1-self.alpha)*self.diff_std  + self.alpha * abs(F - self.u_diff)
        for i in range(bgModel.shape[0]):
            for j in range(bgModel.shape[1]):
                if bgModel[i][j] == 0:
                    self.average_temp[i][j] =u_temp[i][j]# (1-self.alpha)*self.average_temp[i][j] + self.alpha * current_frame[i][j]
                    self.u_diff[i][j] =u_diff_temp[i][j]# (1-self.alpha)*self.u_diff[i][j] + self.alpha*F[i][j]
                    self.diff_std[i][j] =diff_std_temp[i][j]# (1-self.alpha)*self.diff_std[i][j]  + self.alpha * abs(F[i][j]  - self.u_diff[i][j])
    def setBgTemperature(self,avgtemp):
        self.average_temp = avgtemp
    def isCurrentFrameContainHuman(self,current_temp,
            average_temperature,img_diff,show_vote=False):
        '''
            判断当前帧是否含有人类
            ret : (bool,bool) ret[0] True:含有人类，ret[0] False:没有人类，表示属于背景
            ret[1] 为False丢弃这个帧，ret[1]为True，将这个帧作为背景帧
        '''
        ret =  self.knnJudgeFrameContainHuman(current_temp,average_temperature,img_diff,show_vote)
        return ret
    def ensurePathValid(self,customDir):
        if customDir:
            if not os.path.exists(customDir):
                os.mkdir(customDir)
                print("create dir sucessfully: %s" % (customDir))
        else:
            customDir = "imagetemp"
            if not os.path.exists(customDir):
                os.mkdir(customDir)
        return customDir

    def acquireImageData(self,frame_count =1000 ,customDir = None):
        '''
            这个方法是为了获得图像数据，数据保存在customDir中
        '''
        customDir = self.ensurePathValid(customDir)
        # load the avetemp.py stores the average temperature
        # the result of the interpolating for the grid
        average_path = customDir+"/"+"avgtemp.npy"
        print("the average path is %s" % (average_path))
        all_frames = []
        frame_counter = 0  # a counter of frames' num
        # diff_queues saves the difference between average_temp and curr_temp
        try:
            while frame_counter < frame_count:
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
            self.saveImageData(all_frames, customDir)

        except KeyboardInterrupt:
            print("catch keyboard interrupt")
            # save all images
            self.saveImageData(all_frames, customDir)
            print("save all frames")
    def tailOperate(self,currFrame,lastThreeFrame):
        self.countPeopleNum()
        self.showCurrentState()
        #self.updateGaussianBgModel(currFrame)
        self.updateAverageBgModel(currFrame,lastThreeFrame)

    def process(self,testSubDir=None,show_frame=False):
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
            self.preReadPixels()
            print("read sample data ")
            frame_counter = 0 #背景帧数计数器
            seq_counter = 0 #帧数计数器
            bg_frames = [] #保存用于计算背景温度的帧
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
                all_frame.append(currFrame)
                if not self.calcBg:
                    if frame_counter  ==  self.th_bgframes :#是否测完平均温度
                        #更新计算背景的阈值
                        frame_counter=0
                        num = len(bg_frames)
                        print("====num is %d==="%(num))
                        self.average_temp = self.calAverageTemp(bg_frames)
                        self.constructAverageBgModel(bg_frames)
                        bg_frames = [] #清空保存的图片以节省内存
                        print("===average temp is ===")
                        print(self.average_temp)
                        if show_frame:
                            cv.namedWindow("image",cv.WINDOW_NORMAL)
                        self.calcBg = True # has calculated the bg temperature
                    else:
                        frame_counter += 1#帧数计数器自增

                    continue
                last_frame_step = all_frame[ seq_counter - 1 - self.step] 
                print("========================================================process============================================================")
                diff_temp = currFrame - self.average_temp
                if show_frame:
                    plot_img = np.zeros(currFrame.shape,np.uint8)
                    plot_img[np.where(diff_temp > 1.5)] = 255
                    img_resize = cv.resize(plot_img,(16,16),interpolation=cv.INTER_CUBIC)
                    cv.imshow("image",img_resize)
                    cv.waitKey(1)
                ret =self.isCurrentFrameContainHuman(currFrame.copy(),self.average_temp.copy(), diff_temp.copy() )
                if not ret:
                    self.updateObjectTrackDictAgeAndInterval()
                    self.tailOperate(currFrame,last_frame_step)
                    if self.getExistPeople():
                        '''
                        print("============restart calculate the bgtemp======")
                        self.calcBg=False
                        bg_frames = []#重置背景缓冲区
                        self.frame_counter =0 #重置背景帧数计数器
                        '''
                        self.setExistPeople(False)
                    continue
                self.setExistPeople(True)
                print("extractbody")
                (cnt_count,image ,contours,hierarchy),area =self.extractBody(self.average_temp, currFrame)
                if cnt_count ==0:
                    self.updateObjectTrackDictAgeAndInterval()
                    self.tailOperate(currFrame,last_frame_step)
                    continue
                #下一步是计算轮当前帧的中心位置
                loc = self.findBodyLocation(diff_temp,contours,[ i for i in range(self.row)])
                self.trackPeople(currFrame,loc)#检测人体运动轨迹
                self.updateObjectTrackDictAge()#增加目标年龄
                self.tailOperate(currFrame,last_frame_step)
        except KeyboardInterrupt:
            print("catch keyboard interrupt")
            if show_frame:
                cv.destroyAllWindows()
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
            np.save(output_path+"bgmodel.npy",self.bgModel)
            np.save(output_path+"fgmodel.npy",self.fgModel)
            print("sucessfully save the image data")
            print("path is in "+output_path)

    def showCurrentState(self):
        self.showPeopleNum()
        self.showEntranceExitEvents()
        self.showEnterNum()
        self.showExitNum()
    def showEnterNum(self):
        print("======================current enter num is %d================"%(self.__enter))
    def showExitNum(self):
        print("======================exit num is %d================"%(self.__exit))
    def showPeopleNum(self):
        print("=================current people num is %d ==============="%(self.__peoplenum))
    def showEntranceExitEvents(self):
        print("=================current entrance num is %d =================="%(self.__entrance_exit_events))
    def getExistPeople(self):
        return self.__isExist
    def removeNoisePoint(self,diff_temp,corr):
        max_temperature_thresh=2
        small_thresh = 1.5
        horizontal_thresh = 2
        cp_temp_dict = {}
        corr_set = set(corr)
        corr_bak=corr_set.copy()
        for item in corr_set:
            local_max_temp = diff_temp[item]
            if local_max_temp  >=  small_thresh:
                if  self.isDoorHigh or local_max_temp >= max_temperature_thresh:
                    cp_temp_dict[item] = diff_temp[item]
            else:
                corr_bak.remove(item)
        if len(corr_bak) <= 1:
            return list(corr_bak)
        cp_item_sorted =sorted(cp_temp_dict.items(),key =lambda d:d[1],reverse=True)
        cp_set = set(cp_item_sorted)
        reference_set =set()
        removed_set=set()
        for item in cp_item_sorted:
            reference_point=item
            if reference_point in reference_set or reference_point in removed_set:
                continue
            reference_set.add(reference_point)
            rest_points = cp_set - reference_set - removed_set
            for k in rest_points:
                hori_dis = abs(reference_point[0][1] -k[0][1])
                vertical_dis = abs(reference_point[0][0] - k[0][0])
                streetDis = hori_dis +vertical_dis
                if hori_dis <=  horizontal_thresh and not self.isDoorHigh :
                    if streetDis <= 3:
                        removed_set.add(k)
                elif self.isDoorHigh:
                    if streetDis < 3:
                        removed_set.add(k)
        final_corr = []
        rest_set = cp_set-removed_set
        #print("================rest set is ======================")
        #print(rest_set)
        for k,v in rest_set:
            final_corr.append(k)
        return final_corr
    def __splitContours(self,label,contours):
        refindContours = False
        row_index = int(self.row/2)-1
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            if h >= self.row-2:
                label[row_index,x:x+w] = 0
                label[row_index+1,x:x+w]=0
                refindContours = True
        return refindContours
    def findContours(self,img):
        ret =cv.findContours(img,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(ret ) > 2:
            return ret
        return (img,) + ret 
    def __findContours(self,label,key_arr):
        temp = np.zeros((self.row,self.col),np.uint8)
        cnts = []
        heir = None
        for l in key_arr:
            temp[np.where(label == l)] =1
            temp[np.where(label!=l)]=0
            img ,contours,heir=self.findContours(temp)
            cnts.append(contours[0])
        return (len(key_arr),label,cnts,heir)
    def getActualHeightWidth(self,cnt,label):
        x,y,w,h = cv.boundingRect(cnt)
        max_width = 0
        for j in range(y,y+h):
            row  = label[j][x:x+w]
            index = np.where(row !=0 )
            size = len(index[0])
            if size > max_width:
                max_width = size
        return h,max_width
    def __hasTwoPeople(self,h0,w0,ratio,area):
        if ratio >= 1.5 and area >=(self.row-2)*(self.col-4):
            return True
        if ratio >= 2:
            if  h0 >= (self.row -1):
                return True
            elif  h0 >= (self.row-2) and ( area >= 2*(self.col-1)):
                return True
            elif  area >= (self.row-4)*(self.row-4):
                return True
        elif ratio >=3 and h0 >= (self.row-2):
            return True
        return False
    def __getFinalContours(self,label,contours_cache,remove_mask,show_frame=False):
        label[np.where(contours_cache==1)]=1
        #label[np.where(remove_mask>0)] = 0
        if show_frame:
            plt.imshow(label)
            plt.show()
        label = label.astype(np.uint8)
        n,label = cv.connectedComponents(label,connectivity=4)
        return self.__findContours(label,[i for i in range(1,n)]),0
    def bgHighTemperature(self,diff_temp,thre_temp):
        pass
    def extractBody(self,average_temp,curr_temp,show_frame=False,seq=None):
            # issue version
            curr_temp = curr_temp.copy()
            thre_temp =average_temp.copy()+1
            ones = np.ones(average_temp.shape,np.float32)
            diff_test =ones*( curr_temp > thre_temp)
            if diff_test.sum() >= self.image_thresh:
                thre_temp += 1
            iter_count = 0
            max_iter = 2
            single_dog = False
            all_area = self.image_size
            contours_cache = np.zeros((self.row,self.col),np.uint8)
            remove_mask = np.zeros((self.row,self.col),np.uint8)
            while True:
                bin_img = ones*(curr_temp>= thre_temp)
                bin_img = bin_img.astype(np.uint8)
                label=np.zeros((self.row,self.col))
                n , label = cv.connectedComponents(bin_img,label,connectivity=4 )
                iter_count += 1
                area_arr = []
                label_dict= {}
                for i in range(1,n):
                    sub_matrix = label[np.where(label==i)]
                    area_arr.append(sub_matrix.size)
                    label_dict[i] = sub_matrix.size
                    if iter_count == 1 and  sub_matrix.size <= 2 :
                        remove_mask[np.where(label == i)] = i
                if not area_arr:
                    return (0,None,None,None),0
                sorted_label_dict = sorted(label_dict.items(),key=lambda d:d[1],reverse=True)
                max_label_tuple = sorted_label_dict[0]
                max_area = max_label_tuple[1]
                if iter_count  > 0:
                    #print("max_size is %d "%(max_area))
                    if show_frame :
                        plt.imshow(label)
                        plt.show()
                if n == 2 and (max_area <= self.image_thresh):
                    label = label.astype(np.uint8)
                    label , contours,heir=self.findContours(label)
                    h0,w0 = self.getActualHeightWidth(contours[0],label)
                    if h0 <= self.stop_extract_h:#小于等于这个阈值则可以停止迭代过程
                            return self.__getFinalContours(label,contours_cache,remove_mask)
                if iter_count >= max_iter:#超过最大的迭代次数
                    isReturn = False
                    sum_area = sum(area_arr)
                    if iter_count==max_iter :
                        if sum_area <= self.single_people_area:
                            single_dog = True
                        min_item = sorted_label_dict[-1]
                        if min_item[1] == 1:
                            sorted_label_dict.remove(min_item)
                            del label_dict[min_item[0]]#去掉大小为1的连通分量
                            curr_temp[np.where(label==min_item[0])]=0
                    for l ,size in sorted_label_dict:
                        if n > 4:
                            thre_temp += 0.25
                            break
                        if size <= self.col-3 and size > 2:
                            contours_cache[np.where(label==l)] = 1
                        temp = np.zeros((self.row,self.col)).astype(np.uint8)
                        temp[np.where(label ==l)] = 1
                        temp , contours,heir=self.findContours(temp)
                        if not contours:
                            break
                        h0,w0 = self.getActualHeightWidth(contours[0],label)
                        ratio = h0/w0
                        area = w0*h0
                        if  w0 > self.col-4 or max_area >= self.image_size*0.45:
                            thre_temp +=0.25
                            break
                        proportion = self.__hasTwoPeople(h0,w0,ratio,area)
                        if proportion:
                            #print("proportion is true")
                            isReturn=True
                            self.__splitContours(label,contours)
                        else:
                            if size >= 3 and size <= self.save_contours_row:#是否满足保存轮廓的条件
                                contours_cache [np.where(label == l)]=1#这是为了保存之前提取的轮廓
                    if isReturn:
                        print("case 0 ")
                        return self.__getFinalContours(label,contours_cache,remove_mask)
                    if single_dog or  max_area < all_area/8:#尽可能减少高温区域的面积
                        print("==================case 2===========")
                        min_label = sorted_label_dict[-1]
                        if min_label[1]==1 and iter_count <= max_iter:
                            label[np.where(label==min_label[0])]=0
                        return self.__getFinalContours(label,contours_cache,remove_mask)
                elif max_area  < math.ceil(all_area*0.1):#
                    return self.__getFinalContours(label,contours_cache,remove_mask)
                thre_temp += 0.25
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
    def simulateProcess(self,data_path = None,show_frame = False,all_frame = None,avgtemp = None):
        if data_path:
            all_frame = np.load(data_path+"/imagedata.npy")
            avgtemp = np.load(data_path +"/avgtemp.npy")
        self.constructAverageBgModel(all_frame[0:self.M])
        seq_arr = []
        point_arr = []
        diff_frame= []
        if show_frame:
            cv.namedWindow("image",cv.WINDOW_NORMAL)
        for i in range(self.M,all_frame.shape[0]):
            print(" %d frame in all frame "%(i))
            currFrame = all_frame[i]
            last_frame_step = all_frame[i - self.step]
            seq = i
            print(currFrame)
            diff_temp = currFrame - self.average_temp
            if show_frame:
                plot_img = np.zeros(currFrame.shape,np.uint8)
                plot_img[np.where(diff_temp > 1.5)] = 255
                img_resize = cv.resize(plot_img,(16,16),interpolation=cv.INTER_CUBIC)
                cv.imshow("image",img_resize)
                cv.waitKey(30)
            ret =self.isCurrentFrameContainHuman(currFrame.copy(),self.average_temp.copy(), diff_temp.copy() )
            if not ret[0]:
                self.updateObjectTrackDictAgeAndInterval()
                self.tailOperate(currFrame,last_frame_step)
                if self.getExistPeople():
                    self.setExistPeople(False)
                continue
            self.setExistPeople(True)
            print("extractbody")
            (cnt_count,image ,contours,hierarchy),area =self.extractBody(self.average_temp, currFrame)
            if cnt_count ==0:
                self.updateObjectTrackDictAgeAndInterval()
                self.tailOperate(currFrame,last_frame_step)
                continue
            #下一步是计算轮当前帧的中心位置
            loc = self.findBodyLocation(diff_temp,contours,[ i for i in range(self.row)])
            seq_arr.append(seq)
            point_arr.append(loc)
            diff_frame.append(diff_temp)
            self.trackPeople(currFrame,loc)#检测人体运动轨迹
            self.updateObjectTrackDictAge()#增加目标年龄
            self.tailOperate(currFrame,last_frame_step)
        for i in  range(len(seq_arr)):
            seq = seq_arr[i]
            print(seq,end = ":")
            diff = diff_frame[i]
            pos_arr = point_arr[i]
            if len(pos_arr) == 0 :
                print()
                continue
            for p in  pos_arr:
                print(p,end = "=====>")
                print(diff[p[0]][p[1]],end = " ; ")
            print()
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

    def findBodyLocation(self,img,contours):
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
            if temperature != max_temp:
                xcorr,ycorr = np.where(mask == temperature)
                row = xcorr[0]+y
                col = ycorr[0]+x
            corr.append((row,col))
        ret =  self.removeNoisePoint(img,corr)
        return ret
    def showContours(self,img,contours):
        print("====================now paint the contours of the image========")
        img2 = np.array(img.copy(),np.uint8)
        cv.drawContours(img2,contours ,-1 ,(0,255,0),1)
        cv.imshow("contours",img2)
        cv.waitKey(1)
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
    def __inLineWithCurrentSportTrend(self,direction,cp,last_place):
        if direction == 1:#分析运动趋势，如果方向为正，当前坐标一般大于前一个坐标
            if cp[1] < last_place[1]:
                return False
        elif cp[1] > last_place[1]:
            return False
        return True
    def __double_check_corr(self,hozi,verti):
        if hozi < self.hozi_double_check_thresh and verti < self.verti_double_check_thresh:
            return True
        street_dis = hozi + verti
        if street_dis <= self.__street_dis_thresh:
            if hozi <= self.hozi_double_check_thresh:#如果街区
                return True
        return False

    def __extractFeature(self,img,corr):
        '''
        为当前帧提取目标特征
        '''
        #空间距离
        print("==================extract feature=========================")
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
                   x_dis = abs(last_place[1] - corr[0][1])
                   y_dis = abs(last_place[0] - corr[0][0])
                   if self.__double_check_corr(x_dis,y_dis):
                       v.put(corr[0],img)
                       v.clearInterval()
                       return 
        obj_set =set()
        for cp in corr:
            for k,v in self.__objectTrackDict.items():
                obj_set.add(k)
                last_place ,last_frame = v.get()
                #温度差不会超过某个阈值
                previous_img = last_frame
                previous_cnt = previous_img[last_place[0],last_place[1]]#前一帧的中心温度
                diff_temp = abs(img[cp[0],cp[1]] - previous_cnt)#当前中心温度和前一帧数的中心温度差
                horizontal_dis =abs(cp[1] - last_place[1])
                vertical_dis = abs(cp[0] - last_place[0])
                if vertical_dis <= self.__x_thresh  and horizontal_dis <= self.__y_thresh :
                    if  k not in updated_obj_set and cp not in removed_point_set:#防止重复更新某些目标的点a
                        direction = v.getLastTrend()
                        if not self.__inLineWithCurrentSportTrend(direction,cp,last_place):
                            continue
                        self.__objectTrackDict[k].put(cp,img)
                        updated_obj_set.add(k)
                        removed_point_set.add(cp)
        point_rest = set(corr) - removed_point_set
        obj_length = len(updated_obj_set)
        obj_rest = obj_set - updated_obj_set#剩余的未被更新轨迹的对象
        final_point_rest= point_rest.copy()#最终剩余的点，表示新进入的目标，位于视野边缘
        final_obj_rest = obj_rest.copy()#最终剩余的目标，表示该目标消失，通过了监控区域
        if obj_length < obj_num:#是否还有目标尚未匹配
            for point in point_rest :
                for obj in obj_rest :
                    if obj not in final_obj_rest or point not in final_point_rest :#如果目标已经更新了轨迹
                        continue
                    v = self.__objectTrackDict[obj]
                    prev_point , prev_img = v.get()
                    diff_temp = abs(prev_img[prev_point[0],prev_point[1]] - img[point[0],point[1]])
                    hozi_dis = abs(prev_point[1]-point[1])
                    verti_dis = abs(prev_point[0]-point[0])
                    if self.__double_check_corr(hozi_dis,verti_dis):
                        if hozi_dis > 1 :
                            direction = v.getLastTrend()#得到本目标的运动趋向
                            if not self.__inLineWithCurrentSportTrend(direction,point,last_place):#分析运动趋势
                                continue
                        self.__objectTrackDict[obj].put(point,img)
                        final_point_rest.remove(point)
                        final_obj_rest.remove(obj)
        if len(final_point_rest)> 0:#是否有新的人进入监控视野
            for point in final_point_rest:
                if self.nearlyCloseToEdge(point):
                    obj = Target()
                    v = ObjectTrack()
                    v.put(point,img)
                    v.clearInterval()
                    v.showContent()
                    self.__objectTrackDict[obj]= v
                else:
                    print("discard the point (%d,%d)"%(point[0],point[1]))
        if len(final_obj_rest) > 0 :#证明有些人可能已经通过监控区域，或者由于误差没在这个帧出现
            otd = self.__objectTrackDict
            for k in final_obj_rest:
                otd[k].incrementInterval()
            self.updateSpecifiedTarget(final_obj_rest)
    def nearlyCloseToEdge(self,point):#近似作为边界
        if point[1]>= self.col-2 or point[1]-1<= 0:
            return True
        return False
    def updateObjectTrackDictAgeAndInterval(self):
        otd = self.__objectTrackDict
        for k,v in otd.items():
            v.incrementAge()
            v.incrementInterval()
    def updateObjectTrackDictAge(self):
        for k,v in self.__objectTrackDict.items():
            v.incrementAge()
    def updateObjectTrackDictInterval(self):
        for k,v in self.__objectTrackDict.items():
            v.incrementInterval()
    def countPeopleNum(self):
        keys  = self.__objectTrackDict.keys()
        self.updateSpecifiedTarget(keys)

    def updateSpecifiedTarget(self,key):#某个目标突然消失，表示通过监控区域
        removed_set =[]
        for k in key:
            track = self.__objectTrackDict[k]
            if track.isIntervalOverflow() or track.isAgeOverflow():
                if track.hasPassDoor(self.col):
                    self.__entrance_exit_events += 1
                    if track.getDirection() == 1:
                        self.__peoplenum += 1
                        self.__enter += 1
                    else:
                        self.__peoplenum -= 1
                        self.__exit += 1
                removed_set.append(k)
        for k in removed_set:
            del self.__objectTrackDict[k]
    def belongToEdge(self,point):
        if point[1] == 0 or point[1] == self.col-1:
            return True
        return False
    def __classifyObject(self,img, corr):

        '''
        分类目标，确定当前点属于哪个目标

        '''
        print("======classifyObjec=====")
        self.__extractFeature(img,corr)
        #iself.showTargetFeature()

    def showTargetFeature(self):
        print("====show target feature===")

        for k,v in self.__objectTrackDict.items():
            print(k,end=",")
            v.showContent()
        print("")
    def trackPeople(self,img , loc):
        '''
        loc:对象当前位置数组
        img:图片
        '''
        self.__classifyObject(img,loc)
if __name__ == "__main__":
    if len(sys.argv) > 1 :
        if  sys.argv[1]=="process" or sys.argv[1]=="simulate":
            cp = CountPeople(row = 8 ,col=8)
            outputSubDir=None
            if len(sys.argv) > 2:
                outputSubDir =  sys.argv[2]
            show_frame= False
            if len(sys.argv) > 3:
                if sys.argv[3] == "show_frame":
                    show_frame=True
            if sys.argv[1]== "process":
                print("use process deal frame")
                cp.process(outputSubDir,show_frame=show_frame)
            else:
                print("simulate ")
                cp.simulateProcess(outputSubDir,show_frame = show_frame)
        elif sys.argv[1] == "collect":
            if len(sys.argv)>2:
                subdir =""
                if sys.argv[2] == "delay":
                    print('delay 10 s')
                    time.sleep(10)
                    # sys.argv[2] represents the custom  dir of  the image saved    
                    if len(sys.argv) > 3:
                        subdir = sys.argv[3]
                else:
                    subdir = sys.argv[2]
                current_dir = os.path.abspath(os.path.dirname(__file__))
                adjust_dir = current_dir
                if current_dir.endswith("grideye"):
                    adjust_dir = current_dir + "/countpeople"
                packageDir = adjust_dir
                actual_dir = adjust_dir
                path_arg = ""
                actual_dir = actual_dir + "/"+subdir
                # sleep 10 s
                print("the actual_dir is %s" % (actual_dir))

                if not os.path.exists(actual_dir):
                    os.mkdir(actual_dir)
                countp = CountPeople()
                countp.preReadPixels()
                # 这是为了方便访问背景温度数据而保存了包countpeople的绝对路径
                countp.setPackageDir(packageDir)
                try:
                    countp.acquireImageData(customDir=actual_dir)
                except KeyboardInterrupt("keyboard interrupt"):
                    print("exit")
