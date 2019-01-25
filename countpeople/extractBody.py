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
cp = CountPeople(row=8,col=8)
average_median = cp.medianFilter(average_temp_intepol)
average_median_unintel = cp.medianFilter(average_temp)
print("create countpeople object")
def showImage(original , newImage):
    figure , (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(original)
    ax1.set_title("original image")
    ax2.imshow(newImage)
    ax2.set_title("image contours")
    plt.show()
all_result = []
for i in range(sel_frames.shape[0]):
    print("the %dth frame in all_frames "%(frame_arr[i]))
    #frame = all_frames_intepol[i]
    frame = sel_frames[i]
    frame_copy = frame.copy()
    medianBlur = cp.medianFilter(frame)
    print("after the median filter")
    print(medianBlur)
    print("capture the body contours")
    cnt_count , img2,contours , hierarchy = cp.extractBody(average_median_unintel , medianBlur)
    if cnt_count == 0:
        print("current frame has no people")
        continue
    print("return image is ")
    print(img2)
    all_result.append(cnt_count)
    print("==============================has %d people in this frame======================= "%(cnt_count))
    if cnt_count > 0:
        pos = cp.findBodyLocation(medianBlur,contours,[i for i in range(cp.row)])
        for item in pos:
    #        print("=================body is on the place (%d,%d) of the frame ======================"%(item[0],item[1]))
            print(item)
        time.sleep(2)

    showImage(frame_copy , img2)
print(all_result)



