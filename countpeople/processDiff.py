import numpy as np
import os
import sys
from interpolate import imageInterpolate
from countpeople import CountPeople

def analyseImageData(imagedata,avgtemp,cp=None, interpolate_method="linear",end=-1):
    minArr,maxArr,average=[],[],[]
    #imagedata = imageInterpolate(imagedata,interpolate_method)
    #avgtemp = imageInterpolate(avgtemp,interpolate_method)
    diff_queues = []
    if end <= 0  :
        end = len(imagedata)
    for i in range(end):
        diff_queues.append(imagedata[i] - avgtemp)
    diff_queues = np.array(diff_queues)
    diff_queues = np.round(diff_queues,2)
    #print("the length of all images is %d"%(len(diff_queues)))
    for item in diff_queues:
        minArr.append(item.min())
        maxArr.append(item.max())
        average.append(np.average(item))
    minArr,maxArr,average = np.array((minArr,maxArr,average))
    minm = minArr.min()
    maxm = maxArr.max()
    aveave = np.average(average)
    overThresh = []
    for i in range(len(maxArr)):
        current_temp,diff = imagedata[i],diff_queues[i]
        if maxArr[i].max() >2.7:
            overThresh.append(i)
        elif cp:
            ret = cp.isCurrentFrameContainHuman(current_temp,avgtemp,diff)
            if ret[0]:
                overThresh.append(i)

    print("the min array are as listed")
    print(minArr)
    print("===============================the minimum value is %.1f======================"%(minm))
    print("the index of min value in minArr is")
    print(np.where(minArr == minm))
    print("the max array are as listed")
    print(maxArr)
    print("the maximum value is %.1f"%(maxm))
    print("the index of max value in maxArr is")
    print(np.where(maxArr == maxm))
    print("the average values are as listed")
    print(average)
    print("the average value is %.2f"%(aveave))
    return (minm,maxm,aveave,overThresh,imagedata.shape[0])

if __name__ == "__main__":
    if len(sys.argv) <2 :
        raise ValueError("please speciffied the input file")
    path = sys.argv[1]
    if not os.path.exists(path):
        raise ValueError("the path is invalid")
    cp = CountPeople()
    allframe = np.load(path+"/imagedata.npy")
    avgtemp  = np.load(path+"/avgtemp.npy")
    end = -1
    if len(sys.argv) > 2:
        end = int(sys.argv[2])
    ret = analyseImageData(allframe,avgtemp,cp=cp,end = end)
    print("the index of the frame is over thresh is as listed")
    overThresh = ret[3]
    np.save(path+"/human_data.npy",np.array(overThresh))#保存超过阈值的帧序号
    for item in overThresh:
        print(item,end=" ")
    print()
