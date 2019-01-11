import numpy as np
import os
import sys
if len(sys.argv) <2 :
    raise ValueError("please speciffied the input file")
path = sys.argv[1]
if not os.path.exists(path):
    raise ValueError("the path is invalid")
minArr,maxArr,average=[],[],[]
allImage = np.load(path)
print("the length of all images is %d"%(len(allImage)))
for item in allImage:
    item = np.array(item)
    minArr.append(item.min())
    maxArr.append(item.max())
    average.append(np.average(item))
minArr,maxArr,average = np.array((minArr,maxArr,average))
print("the min array are as listed")
print(minArr)
print("the minimum value is %.1f"%(minArr.min()))
print("the max array are as listed")
print(maxArr)
print("the maximum value is %.1f"%(maxArr.max()))
print("the average values are as listed")
print(average)
print("the average value is %.2f"%(np.average(average)))






