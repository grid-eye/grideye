import numpy as np
import os
import sys
import matplotlib.pyplot as plt
imagePrefix="images"
bgPrefix="imagedata"
patharg = sys.argv[1]
framePath = imagePrefix+"/"+patharg+"/imagedata.npy"
diffPath = imagePrefix+"/"+patharg+"/diffdata.npy"
avePath = bgPrefix+"/"+patharg+"/avgtemp.npy"
allframe = np.load(framePath)
diff = np.load(diffPath)
avetemp = np.load(avePath)
argarr = []
for i in range(2, len(sys.argv)):
    argarr.append(int(sys.argv[i]))
j = 1 
print("the avgarr's average is %.2f"%(np.mean(avetemp)))
for i in argarr:
    print("for the %dth frame"%(i))
    print("the max diff is %.2f"%(diff[i].max()))
    print("the average of the frame is %2.f"%(np.mean(allframe[i])))
    ave_diff =  np.mean(allframe[i]) - np.mean(avetemp)
    print("the average's diff between the average and current temp are %.2f"%(ave_diff))
    plt.figure(num=i)
    plt.subplot(2,1,1)
    plt.title("%dth frame"%(i))
    plt.imshow(allframe[i])
    plt.tight_layout()
    plt.xticks([]),plt.yticks([])
plt.show()
    


