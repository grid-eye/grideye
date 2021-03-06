import numpy as np
from processDiff import analyseImageData
from countpeople import CountPeople
import os
import sys
if len(sys.argv) < 2:
    raise ValueError("please speciffiy the path")
num = 1
if len(sys.argv) >=  3:
    num = int(sys.argv[2])

path = sys.argv[1]
if os.path.exists(path) == False:
    raise ValueError("please input a valid path")
path = path[0:len(path)-1]
maxArr,minArr,aveArr,overThresh =[],[],[],[]
cp = CountPeople()
frame_sums = 0
end = 0
if len(sys.argv) > 3:
    end = int(sys.argv[3])
print("==============end = %d =========="%(end))
avgtemp = None
loadAvg = False
for i in range(num):
    realpath = path+str(i+1)
    if os.path.exists(realpath) == False:
        print("this frame sequence doesn't exist")
        continue
    print(" the %dth sequence "%(i))
    if end == 0 :
        end = -1
    allframe = np.load(realpath+"/imagedata.npy")
    if not loadAvg:
        loadAvg = True
        print("==========only load avgtemp  one time=============")
        avgtemp = np.load(realpath+"/avgtemp.npy")
    ret = analyseImageData(allframe,avgtemp,end=end)
    minArr.append(ret[0])
    maxArr.append(ret[1])
    aveArr.append(ret[2])
    overThresh.append(ret[3])
    frame_sums += ret[4]
print("we sucessfully analyse all image data")
print("the maximum of the diff data is %.2f"%(np.array(maxArr).max()))
print("the minimum of the diff data is %.2f"%(np.array(minArr).min()))
print("the average of the all diff data is %.1f"%(np.average(np.array(aveArr))))
print("the array of the frame which is over thresh is as list:")
print("the frame sums is %d "%(frame_sums))
print(np.array(overThresh))
over_sum = 0
for item in overThresh:
    over_sum += len(item)
print("================sum of the overThresh is===================")
print(over_sum)
overIndex = []
for i in range(len(overThresh)):
    if len(overThresh[i]) > 1:
        overIndex.append(i)
print(np.array(overIndex))

