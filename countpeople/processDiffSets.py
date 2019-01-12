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
maxArr,minArr,aveArr =[],[],[]
for i in range(num):
    realpath = path+str(i+1)+"/diffdata.npy"
    ret = analyseImageData(realpath)
    minArr.append(ret[0])
    maxArr.append(ret[1])
    aveArr.append(ret[2])
print("we sucessfully analyse all image data")
print("the maximum of the diff data is %.2f"%(np.array(maxArr).max()))
print("the minimum of the diff data is %.2f"%(np.array(minArr).min()))
print("the average of the all diff data is %.1f"%(np.average(np.array(aveArr))))


