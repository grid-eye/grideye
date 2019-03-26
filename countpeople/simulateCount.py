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
def analyseFrameSequence(frame_arr,all_frames,average_temp,show_frame=False,show_extract_frame=False):
    select_frames_dict = {}
    select_frames_list = []
    area_ret= []#返回的结果，表示人经过监控区域经过初步阈值处理后的面积，用于判断经过的人数
    for i in frame_arr:
        select_frames_dict[i] = all_frames[i]
        select_frames_list.append(all_frames[i])
    sel_frames = np.array(select_frames_list , np.float32)
    target_frames = sel_frames
    cp = CountPeople(row=8,col=8)
    average_median = cp.gaussianFilter(average_temp)
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
        frame =  target_frames[i]
        blur =frame #cp.gaussianFilter(frame)
        seq = frame_arr[i]#表示选择的帧的序号，不一定从0开始
        curr_diff= blur - average_median
        start_time = time.perf_counter()
        ret = cp.isCurrentFrameContainHuman(blur.copy(),average_median.copy(),curr_diff.copy())
        end_time = time.perf_counter()
        interval = end_time - start_time
        #print("=============analyse this frame contain human's time is====================")
        #print(interval)
        if not ret[0]:
            cp.updateObjectTrackDictAgeAndInterval()
            cp.countPeopleNum()
            cp.showCurrentState()
            if cp.getExistPeople():
                cp.setExistPeople(False)
            continue
        cp.setExistPeople(True)
        print("capture the body contours")
        start_time = time.perf_counter()
        (cnt_count , img2,contours , hierarchy),area = cp.extractBody(average_median , blur,show_extract_frame)
        end_time = time.perf_counter()
        interval = end_time - start_time
        #print("=====================extractBody's time is====================")
        #print(interval)
        area_ret.append(area)
        if cnt_count == 0:
            print("current frame has no people")
            cp.updateObjectTrackDictAgeAndInterval()
            cp.countPeopleNum()
            cp.showCurrentState()
            continue
        plt_frames.append(seq)
        rect_arr = []
        for cont in contours:
            x,y,w,d = cv.boundingRect(cont)
            rect_arr.append((x,y,w,d))
        all_result.append(cnt_count)
        contours_rect.append(rect_arr)
        curr_arr.append(curr_diff)
        diff_ave_curr =  curr_diff
        start_time = time.perf_counter()
        pos = cp.findBodyLocation(diff_ave_curr,contours,[i for i in range(cp.row)])
        end_time = time.perf_counter()
        interval = end_time -start_time
        mask = np.zeros((cp.row,cp.col),np.uint8)
        for item in pos:
            mask[item[0],item[1]] = 1
        mask_arr.append(mask)
        respect_img.append(blur)
        center_temp_arr.append(pos)
        start_time = time.perf_counter()
        cp.trackPeople(blur,pos)
        end_time = time.perf_counter()
        interval = end_time - start_time
        cp.updateObjectTrackDictAge()
        cp.countPeopleNum()
        cp.showCurrentState()
    result = cp.getPeopleNum()
    mask_arr = np.array(mask_arr)
    respect_img =np.array(respect_img)
    last_seq = 0
    interval = 20
    artificial_count = -1
    for i in  range(len(center_temp_arr)):
        img = curr_arr[i]
        seq = plt_frames[i]
        print(seq,end=",")
        if center_temp_arr[i]:
            if seq > last_seq +interval:
                artificial_count += 1
                print("=============artificial count is %d==============="%(artificial_count))
            last_seq = seq
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
    print("=================entrance exit event is================")
    print(cp.getEntranceExitEvents())
    print("=================artificial count is ===================")
    print(artificial_count)
    y = "n"
    if show_frame == True and len(plt_frames)<=10:
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
    show_img = False
    if len(sys.argv ) > 2:
        frame_arr =[int(i) for i in  sys.argv[2:] ]
        y = input("show image ?y or n:\n")
        if y == "y":
            show_img=True
    else:
        frame_arr = [i for i in range(len(all_frames))]
    analyseFrameSequence(frame_arr,all_frames,average_temp,True,show_img)
