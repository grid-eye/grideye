import numpy as np
import os
import sys
import matplotlib.pyplot as plt
patharg = sys.argv[1]
framePath = patharg+"/imagedata.npy"
avePath = patharg+"/avgtemp.npy"
allframe = np.load(framePath)
avetemp = np.load(avePath)
diff_queues = []
for i in range(len(allframe)):
    diff_queues.append(allframe[i] - avetemp)
diff_queues = np.array(diff_queues)
diff_queues = np.round(diff_queues,1)
argarr = []
for i in range(2, len(sys.argv)):
    argarr.append(int(sys.argv[i]))
j = 1 
print("the avgarr's average is %.2f"%(np.mean(avetemp)))
for i in argarr:
    print("for the %dth frame"%(i))
    print("the max diff is %.2f"%(diff_queues[i].max()))
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
    


