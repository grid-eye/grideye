import sys
import numpy as np
import os
from testHistAccuracy import analyseHistAccuracy
path = sys.argv[1]
bg = True
if sys.argv[2] == "human":
    bg= False
start ,end = [ int(s) for  s in sys.argv[3:]]
fg_sum ,bg_sum ,length_sum = 0,0,0
for i in range(start ,end):
    actual_path = path+str(i)
    print("==========analyse the %sth sequence ========"%(i))
    allframe = np.load(actual_path+"/imagedata.npy")
    avgtemp = np.load(actual_path+"/avgtemp.npy")
    human_path = path+"/human_data.npy")
    pos_data = []
    if os.path.exists(human_path)
        human_data = np.load(human_path)
        for i in human_data:
            pos_data.append(allframe[i])
    if not bg :
        if len(pos_data) > 10:
            allframe = np.array(pos_data)
    length = allframe.shape[0]
    print("===========current length of all frame is %d============"%(length))
    fg_num,bg_num = analyseHistAccuracy(allframe,avgtemp)
    print("========current fg num is %d=========="%(fg_num))
    print("========current bg num is %d=========="%(bg_num))
    fg_sum += fg_num
    bg_sum += bg_num
    length_sum += length
print("===============length_sum is %d ==================="%(length_sum))
print("===============fg_sum is %d ==================="%(fg_sum))
print("===============bg_sum is %d ==================="%(bg_sum))
print("==============fg proportion is %.3.f ==============="%(fg_sum / length_sum))
print("==============bg proportion is %.3.f ==============="%(bg_sum / length_sum))

 






