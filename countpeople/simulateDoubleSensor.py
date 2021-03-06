import sys
import numpy as np
import time
import os
import cv2 as cv
from countpeople import CountPeople
def showData(data):
    for item in data:
        print(np.array(item))
    print("================")

i = 0 
thresh = 80
merge_shape = (15,8)
def mergeDataBak(t1,t2,cp=None):
    temp = np.zeros(t1.shape)
    print(" t1 shape is")
    print(t1.shape)
    #return np.append(t1[0:4],t2[4:],axis=0)
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
                temp[i][j] = max(t1[i][j],t2[i][j])
    return temp
def mergeData(t1,t2,ave=False):
    shape = merge_shape
    split = 16-shape[0]
    if split == 0:
        np.append(t2,t1,axis =0)
    t1,t2 = t1.copy(),t2.copy()
    row = t1.shape[0]
    sub1 = t1[:split]
    sub2 = t1[-split:]
    temp = np.zeros(sub1.shape)
    for i in range(split):
        for j in range(t1.shape[1]):
            if ave:
                temp[i][j] =round(np.average([ t1[i][j] ,t2[i][j]]),2)
            else:
                temp[i][j] =round(max( t1[i][j] ,t2[i][j]) ,2)
    res = np.append(t2[:-split],temp,axis=0)
    return np.append(res,t1[split:],axis=0)
def mergeDataBak1(t1,t2,cp=None):
    temp = np.zeros(t1.shape)
    print(" t1 shape is")
    print(t1.shape)
    if cp :
        avgtemp = cp.getBgTemperature()
    #return np.append(t1[0:4],t2[4:],axis=0)
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            if cp and (i >2 and  i < 5):
                temp[i][j] =avgtemp[i][j]
            else:
                temp[i][j] = max(t1[i][j],t2[i][j])
    return temp
def saveMergeData(merge,avgtemp,path):
    np.save(path+"/imagedata.npy",merge)
    np.save(path+"/avgtemp.npy",avgtemp)
    print("sucessfully save imagedata.npy in path %s"%(path))
def calComplement(s1,s2):
    return np.round(np.average(s1-s2,axis = 0 ),2)
all_merge_frame = []
cp = CountPeople()
path = sys.argv[1]
show_frame = False
if len(sys.argv) > 2:
    if sys.argv[2] == "show_frame" :
        show_frame = True
sensor1 = np.load(path+"/sensor1.npy")
sensor2 = np.load(path +"/sensor2.npy")
print(sensor1.shape)
print(sensor2.shape)
counter = 0 
merge_data= [ ]
last_three = thresh - 3
container = []
print_trible_tuple =[]
complement_arr = []
complement =np.zeros((8,8))
complement.fill(2)
s1_avgtemp = None
s2_avgtemp = None
s2_arr = []
try:
    cp.setRow(merge_shape[0])
    cp.setCol(merge_shape[1])
    complement = calComplement(sensor1[0:thresh],sensor2[0:thresh])
    print("complement is ")
    print(complement)
    np.save(path+"/complement.npy",complement)
    for i in range(sensor1.shape[0]):
        s1 = sensor1[i]
        s2 = sensor2[i]
        counter += 1
        print(" the %dth frame "%(counter))
        s2 = s2 + complement#加上补偿值
        ave = False
        if cp.isCalcBg():
            ret_1 = cp.isCurrentFrameContainHuman(s1,s1_avgtemp,s1-s1_avgtemp)
            #if ret_1[0] == False:
            #    s1 = s1_avgtemp.copy()
            ret_2 = cp.isCurrentFrameContainHuman(s2,s2_avgtemp,s2-s2_avgtemp)
            #if ret_2[0] == False:
             #   s2 = s2_avgtemp.copy()
            ave = ret_1[0] ^ ret_2[0]
        current_frame = mergeData(s1,s2,ave)#合并两个传感器的数据,取最大值
        print("current frame's shape is ")
        print(current_frame.shape)
        merge_data.append(current_frame)
        container.append(current_frame)
        if len(container) > 3:
            last_three_frame = container.pop(0)
        print(np.round(current_frame,2))
        if not cp.isCalcBg(): 
            if i == thresh-1 :
                s1_avgtemp = cp.calAverageTemp(sensor1[0:thresh])
                s2_avgtemp = cp.calAverageTemp(s2_arr)
                avgtemp =cp.constructAverageBgModel(all_merge_frame)
                cp.setCalcBg(True)
                print("==========after construct bg model==========")
                print(np.round(avgtemp,2))
                if show_frame:
                    cv.namedWindow("merge_data",cv.WINDOW_NORMAL)
                    cv.namedWindow("sensor1_data",cv.WINDOW_NORMAL)
                    cv.namedWindow("sensor2_data",cv.WINDOW_NORMAL)
                cp.calcBg = True
                all_merge_frame=[]
                print(s1_avgtemp)
                print(s2_avgtemp)
            else:
                s2_arr.append(s2)
                all_merge_frame.append(current_frame)
            continue
        print(current_frame.shape)
        print(avgtemp.shape)
        diff = current_frame - avgtemp
        #diff =  diff * weight_matrix#乘上权重矩阵
        #current_frame = avgtemp + diff #更新当前帧
        diff_bak = diff
        if show_frame:
            t = 2
            plot_img = np.zeros(current_frame.shape,np.uint8)
            plot_img[ np.where(diff > 1.5) ] = 255
            print(plot_img.shape)
            img_resize  = cv.resize(plot_img,(plot_img.shape[1]*3,plot_img.shape[0]*3),interpolation=cv.INTER_CUBIC)
            cv.imshow("merge_data",img_resize)
            cv.waitKey(t)
            plot_img.fill(0)
            diff = s1 - s1_avgtemp
            plot_img = np.zeros(s1.shape,np.uint8)
            print(plot_img.shape)
            shape = (plot_img.shape[0]*4,plot_img.shape[1]*4)
            plot_img[ np.where(diff > 1) ] = 255
            img_resize  = cv.resize(plot_img,shape,interpolation=cv.INTER_CUBIC)
            cv.imshow("sensor1_data",img_resize)
            cv.waitKey(t)
            plot_img.fill(0)
            diff = s2 - s2_avgtemp
            print(plot_img.shape)
            plot_img[ np.where(diff > 1) ] = 255
            img_resize  = cv.resize(plot_img,shape,interpolation=cv.INTER_CUBIC)
            cv.imshow("sensor2_data",img_resize)
            cv.waitKey(t)
        diff = diff_bak
        print("judge if current frame has people")
        ret = cp.isCurrentFrameContainHuman(current_frame,avgtemp,diff)
        last_three += 1
        if not ret[0]:
            cp.updateObjectTrackDictAgeAndInterval()
            cp.tailOperate(current_frame,last_three_frame)
            if cp.getExistPeople():
                cp.setExistPeople(False)
            continue
        cp.setExistPeople(True)
        print("extract body")
        (cnt_count,image ,contours,hierarchy),area =cp.extractBody(cp.average_temp, current_frame)
        if cnt_count ==0:
            cp.updateObjectTrackDictAgeAndInterval()
            cp.tailOperate(current_frame,last_three_frame)
            continue
        #下一步是计算轮当前帧的中心位置
        print("find body location")
        loc = cp.findBodyLocation(diff,contours,[ i for i in range(cp.row)])
        print_trible_tuple.append((i,diff,loc))
        cp.trackPeople(current_frame,loc)#检测人体运动轨迹
        cp.showTargetFeature()
        cp.updateObjectTrackDictAge()#增加目标年龄
        cp.tailOperate(current_frame,last_three_frame)
        #sleep(0.5)
    for i,frame,loc in print_trible_tuple:
        print("%d:  "%(i),end = "")
        for p in loc:
            print(p,end="==>")
            print(round(frame[p],2),end=",")
        print()
    #saveMergeData(np.array(merge_data),cp.getBgTemperature(),path)
    cp.showCurrentState()
except KeyboardInterrupt:
    print("keyboardInterrupt")
