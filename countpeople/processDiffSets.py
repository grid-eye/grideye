import numpy as np
from processDiff import analyseImageData
import os
import sys
if len(sys.argv) < 2:
    raise ValueError("please speciffiy the path")
num = 1
if len(sys.argv) == 3:
    num = int(sys.argv[2])
path = sys.argv[1]
if os.path.exists(path) == False:
    raise ValueError("please input a valid path")
path = path[0:len(path)-1]
maxArr,minArr,aveArr,overThresh =[],[],[],[]
for i in range(num):

    realpath = path+str(i+1)
    if os.path.exists(realpath) == False:
        print("this frame sequence doesn't exist")
        continue
    ret = analyseImageData(realpath)
    minArr.append(ret[0])
    maxArr.append(ret[1])
    aveArr.append(ret[2])
    overThresh.append(ret[3])
print("we sucessfully analyse all image data")
print("the maximum of the diff data is %.2f"%(np.array(maxArr).max()))
print("the minimum of the diff data is %.2f"%(np.array(minArr).min()))
print("the average of the all diff data is %.1f"%(np.average(np.array(aveArr))))
print("the array of the frame which is over thresh is as list:")
print(np.array(overThresh))


