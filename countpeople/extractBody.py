import numpy as np
import matplotlib.pyplot as plt
from countpeople import CountPeople
from interpolate import imageInterpolate
import time
import os
import math
import sys
import cv2 as cv
def plotImage(original ,img,rect , frame_seq):
    figure , (ax1,ax2,ax3) = plt.subplots(1,3,num=frame_seq)
    ax1.imshow(original)
    ax1.set_title("original image")
    ax2.imshow(img)
    ax2.set_title("image contours")
    rect_img = np.zeros(img.shape,np.uint8)
    rect_img = cv.cvtColor(rect_img,cv.COLOR_GRAY2BGR)
    for r in rect:
        cv.rectangle(rect_img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0),1)
    ax3.imshow(rect_img)
    ax3.set_xticks([]),ax3.set_yticks([])
def showImage(original , newImage,contours_arr,plt_frames):
    if len(newImage.shape) == 3:
        for i in range(len(newImage)):
            img = newImage[i]
            omg = original[i]
            rect = contours_arr[i]
            seq = plt_frames[i]
            plotImage(omg, img,rect,seq)
    else:
        plotImage(original,newImage,contours_arr,plt_frames)
def analyseFrameSequence(frame_arr,all_frames,average_temp,show_frame=False):
    select_frames_dict = {}
    select_frames_list = []
    area_ret= []#返回的结果，表示人经过监控区域经过初步阈值处理后的面积，用于判断经过的人数
    for i in frame_arr:
        select_frames_dict[i] = all_frames[i]
        select_frames_list.append(all_frames[i])
    sel_frames = np.array(select_frames_list , np.float32)
    average_temp_intepol = imageInterpolate(average_temp)
    all_frames_intepol = imageInterpolate(sel_frames)
    cp = CountPeople(row=32,col=32)
    average_median = cp.medianFilter(average_temp_intepol)
    average_median_unintel = cp.medianFilter(average_temp)
    print("create countpeople object")
    all_result = []
    mask_arr = []
    respect_img=[]
    contours_rect = []
    center_temp_arr=[]
    curr_arr = []#保存图片的差值(当前帧和背景的差值)
    plt_frames = []#被绘制的帧的序号
    cp.setExistPeople(False)
    for i in range(sel_frames.shape[0]):
        print("the %dth frame in all_frames "%(frame_arr[i]))
        #frame = all_frames_intepol[i]
        frame = all_frames_intepol[i]
        frame_copy = frame.copy()
        blur = cp.medianFilter(frame)
        seq = frame_arr[i]#表示选择的帧的序号，不一定从0开始
        curr_diff= blur - average_median
        start_time = time.perf_counter()
        ret = cp.isCurrentFrameContainHuman(blur.copy(),average_median.copy(),curr_diff.copy())
        end_time = time.perf_counter()
        interval = end_time - start_time
        print("=============analyse this frame contain human's time is====================")
        print(interval)
        if not ret[0]:
            if cp.getExistPeople():
                cp.updatePeopleCount()
                cp.setExistPeople(False)
            else:
                print("===no people===")
            continue
        else:
            cp.setExistPeople(True)
        print("capture the body contours")
        print("===before extractbody sum is===")
        print(np.sum(blur))
        print("====average median sum is===")
        print(np.sum(average_median))
        start_time = time.perf_counter()
        (cnt_count , img2,contours , hierarchy),area = cp.extractBody(average_median , blur)
        end_time = time.perf_counter()
        interval = end_time - start_time
        interval = end_time -start_time
        print("=====================extractBody's time is====================")
        print(interval)
        area_ret.append(area)
        if cnt_count == 0:
            print("current frame has no people")
            #print(len(contours))
            #raise ValueError("no people")
            continue
        plt_frames.append(seq)
        rect_arr = []
        for cont in contours:
            x,y,w,d = cv.boundingRect(cont)
            rect_arr.append((x,y,w,d))
        all_result.append(cnt_count)
        print("==============================has %d people in this frame======================= "%(cnt_count))
        if cnt_count > 0:
            contours_rect.append(rect_arr)
            curr_arr.append(curr_diff)
            diff_ave_curr =  curr_diff
            start_time = time.perf_counter()
            print("=======max temprature of this frame is ==============")
            print(np.max(diff_ave_curr))
            pos = cp.findBodyLocation(diff_ave_curr,contours,[i for i in range(cp.row)])
            end_time = time.perf_counter()
            interval = end_time -start_time
            print("===============analyse findBodyLocation's tiem ================")
            print(interval)
            mask = np.zeros((cp.row,cp.col),np.uint8)
            for item in pos:
                mask[item[0],item[1]] = 1
           # cp.trackPeople(img2,pos)
            mask_arr.append(mask)
            respect_img.append(blur)
            center_temp_arr.append(pos)
            start_time = time.perf_counter()
            cp.trackPeople(blur,pos)
            end_time = time.perf_counter()
            interval = end_time - start_time
            print("===============analyse track people's time==================")
            print(interval)
    cp.updatePeopleCount()
    result = cp.getPeopleNum()
    print("there are %d people in the room"%(result))
    mask_arr = np.array(mask_arr)
    respect_img =np.array(respect_img)
    print("====print the loc of all center points===")
    for i in  range(len(center_temp_arr)):
        img = curr_arr[i]
        seq = plt_frames[i]
        print(seq,end=",")
        for pos in center_temp_arr[i]:
            print(pos,end="===>")
            print(round(img[pos[0],pos[1]],2) ,end=",")
        print()
    print()
    print("===================calculate the distance===================================")
    pos_arr = []
    for i in center_temp_arr:
        for pos in i:
            pos_arr.append(pos)
    print(pos_arr)
    if len(pos_arr) > 0:
        pre = pos_arr[0]
        for i in range(1,len(pos_arr)):
            pos = pos_arr[i]
            eu_dis = math.sqrt(math.pow(pos[0]-pre[0],2)+math.pow(pos[1]-pre[1],2))
            pre = pos
            print(round(eu_dis,2),end=";")
    y = "n"
    if show_frame == True:
        y = input("plot the img?yes:y,no:enter")
    if y == "y":
        print(plt_frames)
        showImage(respect_img,mask_arr ,contours_rect,plt_frames)
        plt.show()
    return area_ret,cp.getPeopleNum()
if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise ValueError("please specify a valid path and frame array")
    path = sys.argv[1]
    all_frames = np.load(path+"/imagedata.npy")
    average_temp = np.load(path+"/avgtemp.npy")
    print("==============loaded people =====================")
    if len(sys.argv ) > 2:
        frame_arr =[int(i) for i in  sys.argv[2:] ]
    else:
        frame_arr = [i for i in range(len(all_frames))]
    analyseFrameSequence(frame_arr,all_frames,average_temp,True)
