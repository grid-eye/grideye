import numpy as np
import os
import sys
from interpolate import imageInterpolate
from countpeople import CountPeople

def analyseImageData(path,cp, interpolate_method="linear"):
    minArr,maxArr,average=[],[],[]
    imagedata =  np.load(path+"/imagedata.npy")
    avgtemp = np.load(path+"/avgtemp.npy")
    #imagedata = imageInterpolate(imagedata,interpolate_method)
    #avgtemp = imageInterpolate(avgtemp,interpolate_method)
    diff_queues = []
    for i in range(len(imagedata)):
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
    print("the minimum value is %.1f"%(minm))
    print("the max array are as listed")
    print(maxArr)
    print("the maximum value is %.1f"%(maxm))
    print("the average values are as listed")
    print(average)
    print("the average value is %.2f"%(aveave))
    return (minm,maxm,aveave,overThresh)

if __name__ == "__main__":
    if len(sys.argv) <2 :
        raise ValueError("please speciffied the input file")
    path = sys.argv[1]
    if not os.path.exists(path):
        raise ValueError("the path is invalid")
    cp = CountPeople()
    ret = analyseImageData(path,cp=cp)
    print("the index of the frame is over thresh is as listed")
    print(np.array(ret[3]))







