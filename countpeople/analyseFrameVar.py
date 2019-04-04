import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
from interpolate import imageInterpolate
import cv2 as cv
from otsuBinarize import otsuThreshold
def analyseSequence(allframe,avgtemp,argarray,show_frame=False,thresh=None ,interpolate_method = "linear"):
    #allframe = imageInterpolate(allframe,interpolate_method)
    #avgtemp = imageInterpolate(avgtemp,interpolate_method)
    var_arr = []
    var_diff_arr=[]
    for i in argarray:
        print("the %d th diff frame "%(i))
        diff = allframe[i] - avgtemp
        origin_frame=allframe[i]
        var = np.var(origin_frame)
        var_diff = np.var(diff)
        var_diff_arr.append(var_diff)
        var_arr.append(var)
        #print("diff ave of curr temp and avgtemp is %.2f"%(diff_ave))
        #print("the maximum of the diff frame is %.2f"%(diff.max()))
        if show_frame:
            hists,bins = np.histogram(diff.ravel() , bins=120 , range=(-6,6) )
            histMap = {}
            bins = bins[:-1]
            rest = diff.size - hists.sum() 
            hists[-1]+=rest
            ret,thre = otsuThreshold(diff , 64)
            for i in range(len(bins)):
                histMap[bins[i]] = hists[i]
            gaussian = cv.GaussianBlur(diff,(5,5),0)
            plt.figure(num=seq)
            plt.subplot(2,2,1)
            plt.imshow(gaussian)
            plt.xticks([]),plt.yticks([])
            plt.subplot(2,2,2)
            plt.plot(bins , hists)
            plt.subplot(2,2,3)
            plt.imshow(thre)
            plt.xticks([])
            plt.yticks([])
            plt.title("otsu binarize")
    if show_frame:
        plt.show()
    return var_arr,var_diff_arr
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
    if show_frame == "y":
        is_show_frame=True
    ret = analyseSequence(allframe,avgtemp , argarray,show_frame=is_show_frame)
    print("max var of var_arr is %.3f"%(max(ret[0])))
    print("min var of var_arr is %.3f"%(min(ret[0])))
    print("max var of var_diff_arr is %.3f"%(max(ret[1])))
    print("min var of var_diff_arr is %.3f"%(min(ret[1])))
