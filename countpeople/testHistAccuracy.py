import sys
import numpy as np
def analyseHistAccuracy(allframe , avgtemp,xthresh= 1.8,ythresh =2):
    fg_num = 0
    bg_num = 0 
    for item in allframe:
        diff = item - avgtemp
        target_corr = np.where(diff > xthresh)
        num = len(target_corr[0])
        if num > ythresh:
            fg_num += 1
        else:
            bg_num += 1
    return fg_num ,bg_num
if __name__ == "main":
    path = sys.argv[1]
    allframe = np.load(path+"/imagedata.npy")
    avgtemp = np.load(path+"/avgtemp.npy")
    print("===========all length is %d=================="%(allframe.shape[0]))
    fg_num ,bg_num = analyseHistAccuracy(allframe,avgtemp)
    print("===============fg_num is %d =================="%(fg_num))
    print("===============bg_num is %d =================="%(bg_num))


