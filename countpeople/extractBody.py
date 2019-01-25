import numpy as np
import matplotlib.pyplot as plt
from countpeople import CountPeople
from interpolate import imageInterpolate
import time
import os
import sys
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
def plotImage(original ,img,figs = [0]):
    figure , (ax1,ax2) = plt.subplots(1,2,num=figs[0])
    figs[0]+=1
    ax1.imshow(original)
    ax1.set_title("original image")
    ax2.imshow(img)
    ax2.set_title("image contours")
def showImage(original , newImage):
    if len(newImage.shape) == 3:
        for i in range(len(newImage)):
            img = newImage[i]
            omg = original[i]
            plotImage(omg, img)
    else:
        plotImage(original,newImage)

all_result = []
mask_arr = []
respect_img=[]
for i in range(sel_frames.shape[0]):
    print("the %dth frame in all_frames "%(frame_arr[i]))
    #frame = all_frames_intepol[i]
    frame = all_frames_intepol[i]
    frame_copy = frame.copy()
    medianBlur = cp.medianFilter(frame)
    curr = medianBlur - average_median
    ret = cp.isCurrentFrameContainHuman(medianBlur,average_median,curr)
    print("after the median filter")
    print(medianBlur)
    print("capture the body contours")
    cnt_count , img2,contours , hierarchy = cp.extractBody(average_median , medianBlur)
    if cnt_count == 0:
        print("current frame has no people")
        continue
    print("return image is ")
    print(img2)
    all_result.append(cnt_count)
    print("==============================has %d people in this frame======================= "%(cnt_count))
    if cnt_count > 0:
        diff_ave_curr = medianBlur - average_median
        pos = cp.findBodyLocation(diff_ave_curr,contours,[i for i in range(cp.row)])
        for item in pos:
         #print("=================body is on the place (%d,%d) of the frame ======================"%(item[0],item[1]))
            print(item)
            mask = np.zeros((cp.row,cp.col),np.uint8)
            mask[item[0],item[1]] = 1
            mask_arr.append(mask)
            respect_img.append(medianBlur)
#        time.sleep(2)

mask_arr = np.array(mask_arr)
respect_img =np.array(respect_img)
showImage(respect_img,mask_arr )
plt.show()

