import numpy as np
import sys
import os
from testMultiDirHistAccuracy import analyseMultiDirHist
thresh = [1.0,1.75,2,2.25,2.5,2.75]
path = sys.argv[1]
if sys.argv[2] == "human":
    bg=False
    s =3
else:
    s=2
    bg= True
start ,end = [int(i) for i in sys.argv[s:]]
result_map = []
avgtemp = np.load(path+sys.argv[s]+"/avgtemp.npy")
for t in thresh:
    print("===========current thresh is %.2f ==========="%(t))
    fg_num , bg_num,all_length = analyseMultiDirHist(path,start,end,bg,avgtemp,xthresh=t)
    result_map.append((t, [fg_num,bg_num,all_length]))
print(result_map)
for k, v in result_map:
    print("===========%.2f======"%(k))
    print("fg_num/all_length = %.2f"%(v[0]/v[2]))
    print("bg_num/all_length = %.2f"%(v[1]/v[2]))
