import numpy as np
import os
import sys
def analyseImageData(path):
    minArr,maxArr,average=[],[],[]
    imagedata =  np.load(path+"/imagedata.npy")
    average =  np.load(path+"/avgtemp.npy")
    diff_queues = []
    for i in range(len(imagedata)):
        diff_queues.append(imagedata[i] - average)
    diff_queues = np.array(diff_queues)
    diff_queues = np.round(diff_queues,1)
    print("the length of all images is %d"%(len(diff_queues)))
    for item in diff_queues:
        item = np.array(item)
        minArr.append(item.min())
        maxArr.append(item.max())
        average.append(np.average(item))
    minArr,maxArr,average = np.array((minArr,maxArr,average))
    minm = minArr.min()
    maxm = maxArr.max()
    aveave = np.average(average)
    overThresh = []
    if maxm > 2.7:
        for i in range(len(maxArr)):
            if maxArr[i].max() >2.7:
                overThresh.append(i)
        print("oh ,over thresh！")
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
    ret = analyseImageData(path)
    print("the index of the frame is over thresh is as listed")
    print(np.array(ret[3]))







