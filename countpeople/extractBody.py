import numpy as np
import matplotlib.pyplot as plt
from countpeople import CountPeople
from interpolate import imageInterpolate
import time
import os
import sys
import cv2 as cv
if len(sys.argv) < 2:
    raise ValueError("please specify a valid path and frame array")
path = sys.argv[1]
frame_arr =[int(i) for i in  sys.argv[2:] ]
all_frames = np.load(path+"/imagedata.npy")
average_temp = np.load(path+"/avgtemp.npy")
select_frames_dict = {}
select_frames_list = []
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
def plotImage(original ,img,rect , figs = [0]):
    figure , (ax1,ax2,ax3) = plt.subplots(1,3,num=figs[0])
    figs[0]+=1
    ax1.imshow(original)
    ax1.set_title("original image")
    ax2.imshow(img)
    ax2.set_title("image contours")
    rect_img = np.zeros(img.shape,np.uint8)
    rect_img = cv.cvtColor(rect_img,cv.COLOR_GRAY2BGR)
    print("plot")
    print(rect)
    for r in rect:
        cv.rectangle(rect_img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0),1)
    ax3.imshow(rect_img)
    ax3.set_xticks([]),ax3.set_yticks([])
def showImage(original , newImage,contours_arr,figs_num=[0]):
    if len(newImage.shape) == 3:
        for i in range(len(newImage)):
            img = newImage[i]
            omg = original[i]
            rect = contours_arr[i]
            plotImage(omg, img,rect,figs_num)
    else:
        plotImage(original,newImage,contours_arr,figs_num)

all_result = []
mask_arr = []
respect_img=[]
contours_rect = []
center_temp_arr=[]
for i in range(sel_frames.shape[0]):
    print("the %dth frame in all_frames "%(frame_arr[i]))
    #frame = all_frames_intepol[i]
    frame = all_frames_intepol[i]
    frame_copy = frame.copy()
    medianBlur = cp.medianFilter(frame)
    print("after the median filter")
    print(medianBlur)
    curr = medianBlur - average_median
    ret = cp.isCurrentFrameContainHuman(medianBlur,average_median,curr)
    print("capture the body contours")
    cnt_count , img2,contours , hierarchy = cp.extractBody(average_median , medianBlur,False)
    if cnt_count == 0:
        print("current frame has no people")
        raise ValueError("no people")
        continue
    rect_arr = []
    for cont in contours:
        x,y,w,d = cv.boundingRect(cont)
        rect_arr.append((x,y,w,d))
    all_result.append(cnt_count)
    print("==============================has %d people in this frame======================= "%(cnt_count))
    if cnt_count > 0:
        contours_rect.append(rect_arr)
        diff_ave_curr = medianBlur - average_median
        pos = cp.findBodyLocation(diff_ave_curr,contours,[i for i in range(cp.row)])
        for item in pos:
            print("=================body is on the place (%d,%d) of the frame ======================"%(item[0],item[1]))
            print(item)
            mask = np.zeros((cp.row,cp.col),np.uint8)
            mask[item[0],item[1]] = 1
       # cp.trackPeople(img2,pos)
        mask_arr.append(mask)
        respect_img.append(medianBlur)
        center_temp_arr.append(pos)
mask_arr = np.array(mask_arr)
respect_img =np.array(respect_img)
print(mask_arr.shape)
print(respect_img.shape)
print("================contours_rect length is %d================="%(len(contours_rect)))
print(center_temp_arr)
for i in  range(len(center_temp_arr)):
    img = respect_img[i]
    print("=================center=================")
    for pos in center_temp_arr[i]:
        print(img[pos[0],pos[1]])

showImage(respect_img,mask_arr ,contours_rect)
plt.show()

