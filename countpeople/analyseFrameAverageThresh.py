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
    diff_frame = []
    curr_frame =[]
    ave_arr =[]
    for i in argarray:
        diff_frame.append(allframe[i] - avgtemp)
        curr_frame.append(allframe[i])
    diff_frame = np.round(np.array(diff_frame),1)
    for i in range(len(argarray)):
        print("the %sth diff frame "%(argarray[i]))

        seq = argarray[i]
        currframe = diff_frame[i]
        img = curr_frame[i]
        exceed_frame = np.where(currframe<0)
        currframe[exceed_frame] = 0
        img_ave = np.average(currframe)
        ave_arr.append(img_ave)
        curr_average = np.average(img)
        avg_average = np.average(avgtemp)
        #print("diff ave of curr temp and avgtemp is %.2f"%(diff_ave))
        #print("the maximum of the diff frame is %.2f"%(currframe.max()))
        '''
        for k, v in  histMap.items():
            if k > 2:
                exceed_sum += v
        print("====exceed sum is %d ===="%(exceed_sum))
        gaussian -=1 
        ret = np.where(gaussian < 0)
        gaussian[ret] = 0
        start = time.perf_counter()        
        end = time.perf_counter()
        occupy_time = end -start
        print("========time occupyed==============")
        #print(occupy_time)
        #print("the sum of thre after otsu %d"%(thre.sum()))
        #print("it conforms %% %.2f"%((thre.sum()/1024)))
        #print("thresh's sum is")
        #print(thre.sum())
        #print("sum is %.2f"%(thre.sum()))
        '''
        if show_frame:
            hists,bins = np.histogram(currframe.ravel() , bins=120 , range=(-6,6) )
            histMap = {}
            bins = bins[:-1]
            rest = currframe.size - hists.sum() 
            hists[-1]+=rest
            ret,thre = otsuThreshold(currframe , 64)
            for i in range(len(bins)):
                histMap[bins[i]] = hists[i]
            gaussian = cv.GaussianBlur(currframe,(5,5),0)
            plt.figure(num=seq)
            plt.subplot(2,2,1)
            plt.imshow(currframe)
            plt.xticks([]),plt.yticks([])
            plt.subplot(2,2,2)
            plt.plot(bins , hists)
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
    if show_frame == "y":
        is_show_frame=True
    analyseSequence(allframe,avgtemp , argarray,show_frame=is_show_frame)
