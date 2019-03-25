import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
import sys
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
def analyseHighTemperatureAverage(frame_arr,all_frames,average_temp,imagesize = (8,8),show_frame=False):
    diff_arr = []#保存图片的差值(当前帧和背景的差值)
    high_temperature_ave_arr =[]
    partion_index = int(math.ceil(imagesize[0]*imagesize[1]*6/7))
    print("partion_index %d "%(partion_index))
    for i in frame_arr:
        print("the %dth frame in all_frames "%(i))
        curr_frame = all_frames[i]
        diff_temp = curr_frame - average_temp
        diff_arr.append(diff_temp)
        ravel = diff_temp.ravel()
        ravel_sorted = sorted(ravel)
        partion = ravel_sorted[partion_index]
        high_temp_region = diff_temp[np.where(diff_temp >= partion)]
        high_temperature = np.average(high_temp_region)
        high_temperature_ave_arr.append(high_temperature)
    return np.array(high_temperature_ave_arr)

if __name__ == "__main__":
    if len(sys.argv)  <2 :
        raise ValueError("please specify a valid path and frame array")
    path = sys.argv[1]
    all_frames = np.load(path+"/imagedata.npy")
    average_temp = np.load(path+"/avgtemp.npy")
    print("==============loaded people =====================")
    if len(sys.argv ) > 2:
        frame_arr =[int(i) for i in  sys.argv[2:] ]
    else:
        frame_arr = [i for i in range(len(all_frames))]
    ret = analyseHighTemperatureAverage(frame_arr,all_frames,average_temp,(8,8),True)
    print("=================min value of the array is %.3f================"%(ret.min()))
    print("=================max value of the array is %.3f================"%(ret.max()))
    interval = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for i in interval:
        sub_array_up = ret[np.where(ret > i)]
        sub_array_down = ret[np.where(ret <=i)]
        print("=================== > %.2f : %d ==================="%(i,sub_array_up.size))
        print("=================== <=%.2f : %d ==================="%(i,sub_array_down.size))

