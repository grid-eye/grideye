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
for i in range(num):
    realpath = path+str(i+1)+"/diffdata.npy"
    analyseImageData(realpath)
