import sys
import numpy as np
import os
from testHistAccuracy import analyseHistAccuracy
if __name__ == "main":
    path = sys.argv[1]
    bg = True
    if sys.argv[2] == "human":
        bg= False
        s = 3
    else:
        s = 2
    start ,end  = [ int(s) for  s in sys.argv[s:]]
    avgtemp = np.load(path+str(start)+"/avgtemp.npy")
    analyseMultiDirHist(path,start,end,avgtemp)
def analyseMultiDirHist(path , start , end,bg,avgtemp ,xthresh=2,ythresh=3,avgtemp):
    fg_sum ,bg_sum ,length_sum = 0,0,0
    for i in range(start,end):
        actual_path = path+str(i)
        if not  os.path.exists(actual_path):
            print("%s not exist"%(actual_path))
            continue
        print("==========analyse the %sth sequence ========"%(i))
        allframe = np.load(actual_path+"/imagedata.npy")
        human_path = actual_path+"/human_data.npy"
        if not bg :
            pos_data = []
            human_data = np.load(human_path)
            for i in human_data:
                pos_data.append(allframe[i])
            allframe = np.array(pos_data)
        length = allframe.shape[0]
        print("===========current length of all frame is %d============"%(length))
        fg_num,bg_num = analyseHistAccuracy(allframe,avgtemp,xthresh=xthresh,ythresh=ythresh)
        print("========current fg num is %d=========="%(fg_num))
        print("========current bg num is %d=========="%(bg_num))
        fg_sum += fg_num
        bg_sum += bg_num
        length_sum += length
    print("===============length_sum is %d ==================="%(length_sum))
    print("===============fg_sum is %d ==================="%(fg_sum))
    print("===============bg_sum is %d ==================="%(bg_sum))
    if length_sum > 0 :
        print("==============fg proportion is %.3f ==============="%(fg_sum / length_sum))
        print("==============bg proportion is %.3f ==============="%(bg_sum / length_sum))
    return fg_sum,bg_sum,length_sum







