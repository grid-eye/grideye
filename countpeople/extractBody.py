import numpy as np
import matplotlib.pyplot as plt
from countpeople import CountPeople
from interpolate import imageInterpolate
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
cp = CountPeople()
average_median = cp.medianFilter(average_temp_intepol)
print("create countpeople object")
def showImage(original , newImage):
    figure , (ax1,ax2,ax3,ax4) = plt.subplots(2,2)
    ax1.imshow(original)
    ax1.title("original image")
    ax2.imshow(newImage)
    ax2.title("image contours")
    plt.show()
for i in range(all_frames_intepol.shape[0]):
    print("the %dth frame in all_frames "%(frame_arr[i]))
    frame = all_frames_intepol[i]
    frame_copy = frame.copy()
    medianBlur = cp.medianFilter(frame)
    print("after the median filter")
    print(medianBlur)
    print("capture the body contours")
    cnt_count , img2,contours , hierarchy = cp.extractBody(average_median , medianBlur)
    showImage(frame_copy , img2)



