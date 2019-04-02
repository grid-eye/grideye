import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
from interpolate import imageInterpolate
import cv2 as cv
from otsuBinarize import otsuThreshold
def analyseSequence(allframe,avgtemp,argarray,show_frame=False,thresh=None ,interpolate_method = "linear"):
    diff_frame = []
    curr_frame =[]
    ave_arr =[]
    for i in argarray:
        diff_frame.append(allframe[i] - avgtemp)
        curr_frame.append(allframe[i])
    diff_frame = np.round(np.array(diff_frame),1)
    for i in range(len(argarray)):
        print("the %sth diff frame "%(argarray[i]))
        print(allframe[argarray[i]])
        seq = argarray[i]
        currframe = diff_frame[i]
        img_ave = np.average(currframe)
        ave_arr.append(img_ave)
        if show_frame:
            hists,bins = np.histogram(currframe.ravel() , bins=120 , range=(-6,6) )
            histMap = {}
            bins = bins[:-1]
            rest = currframe.size - hists.sum() 
            hists[-1]+=rest
            print(hists)
            ret,thre = otsuThreshold(currframe , 64)
            for i in range(len(bins)):
                histMap[bins[i]] = int(hists[i])
            gaussian = cv.GaussianBlur(currframe,(5,5),0)
            plt.figure(num=seq)
            plt.subplot(2,2,1)
            plt.imshow(currframe)
            plt.xticks([]),plt.yticks([])
            ax2 = plt.subplot(2,2,2)
            ax2.set_ylabel("pixel number")
            ax2.set_xlabel("temperature difference")
            ax2.set_title("temperature difference distribution")
            ax2.plot(bins , hists)
            plt.subplot(2,2,3)
            plt.imshow(thre)
            plt.xticks([])
            plt.yticks([])
            plt.title("otsu binarize")
            plt.tight_layout()
    #print(ave_arr)
    max_v = max(ave_arr)
    min_v = min(ave_arr)
    print("====================max value of the ave_arr is %.2f================"%(max_v))
    print("index of max value is")
    print(ave_arr.index(max_v))
    print("====================minimum value of the ave_arr is %.2f================"%(min_v))
    print("index of min value is")
    print(ave_arr.index(min_v))
    ave_array = np.array(ave_arr)
    curr_thresh = 0.375
    if thresh:
        curr_thresh = thresh
    sub_ave_index = np.where(ave_array>=curr_thresh)
    sub_ave = ave_array[sub_ave_index]
    print("====================current thresh is %.3f============"%(curr_thresh))
    print("=========over thresh ===============")
    #print(sub_ave)
    print("============sum of the length is %d============"%(ave_array.size))
    print("==========len of the sub_ave is %d============="%(sub_ave.size))


    if show_frame:
        plt.show()
    return  ave_array.size,sub_ave.size
if __name__ == "__main__":
    path = sys.argv[1]
    if os.path.exists(path) == False:
        raise ValueError("no such path %s"%(path))
    allframe = np.load(path+"/imagedata.npy")
    avgtemp = np.load(path+"/avgtemp.npy")
    show_frame = "n"
    show_frame = sys.argv[2]
    if len(sys.argv) > 3:
        argarray = sys.argv[3:]
        for i in range(len(argarray)):
            argarray[i] = int(argarray[i])
    else:
        argarray = [i for i in range(len(allframe))]
    print(argarray)
    is_show_frame = False
    if show_frame == "show_frame":
        is_show_frame=True
    analyseSequence(allframe,avgtemp , argarray,show_frame=is_show_frame)
