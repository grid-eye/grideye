import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from interpolate import imageInterpolate
import cv2 as cv
from otsuBinarize import otsuThreshold
def analyseSequence(path ,argarray,interpolate_method = "linear"):
    allframe = np.load(path+"/imagedata.npy")
    avgtemp = np.load(path+"/avgtemp.npy")
    allframe = imageInterpolate(allframe,interpolate_method)
    avgtemp = imageInterpolate(avgtemp,interpolate_method)
    diff_frame = []
    for i in allframe:
        diff_frame.append(i - avgtemp)
    diff_frame = np.round(np.array(diff_frame),1)
    for i in argarray:
        print("the %dth diff frame "%(i))
        currframe = diff_frame[i]
        print("the maximum of the diff frame is %.2f"%(currframe.max()))
        plt.figure(num=i)
        plt.subplot(2,2,1)
        plt.imshow(currframe)
        plt.xticks([]),plt.yticks([])
        hists,bins = np.histogram(currframe.ravel() , bins=120 , range=(-6,6) )
        histMap = {}
        bins = bins[:-1]
        rest = currframe.size - hists.sum() 
        hists[-1]+=rest
        for i in range(len(bins)):
            histMap[bins[i]] = hists[i]
        exceed_sum=0#溢出之和
        for k, v in  histMap.items():
            if k > 2:
                exceed_sum += v
        print("====exceed sum is %d ===="%(exceed_sum))
        plt.subplot(2,2,2)
        plt.plot(bins , hists)
        gaussian = cv.GaussianBlur(diff_frame[i],(5,5),0)
        ret,thre = otsuThreshold(gaussian , 1024,thre = 2.4)
        print("the sum of thre after otsu %d"%(thre.sum()))
        print("it conforms %% %.2f"%((thre.sum()/1024)))
        print("thresh's sum is")
        print(thre.sum())
        print("sum is %.2f"%(thre.sum()))
        plt.subplot(2,2,3)
        plt.imshow(thre)
        plt.xticks([])
        plt.yticks([])
        plt.title("otsu binarize")
        plt.tight_layout()

    plt.show()
if __name__ == "__main__":
    if len(sys.argv) < 2 :
        raise ValueError("please specify a or more than one frame")
    path = sys.argv[1]
    if os.path.exists(path) == False:
        raise ValueError("no such path %s"%(path))
    argarray = sys.argv[2:]
    for i in range(len(argarray)):
        argarray[i] = int(argarray[i])
    analyseSequence(path , argarray)
