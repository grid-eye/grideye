import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cv2 as cv
if len(sys.argv) < 2 :
    raise ValueError("please specify a or more than one frame")
path = sys.argv[1]
if os.path.exists(path) == False:
    raise ValueError("no such path %s"%(path))
argarray = sys.argv[2:]

for i in range(len(argarray)):
    argarray[i] = int(argarray[i])
allframe = np.load(path)
for i in argarray:
    print("the %dth diff frame "%(i))
    currframe = allframe[i]
    plt.figure(num=i)
    plt.subplot(2,2,1)
    plt.imshow(currframe)
    plt.xticks([]),plt.yticks([])
    hists,bins = np.histogram(currframe.ravel() , bins=120 , range=(-6,6) )
    bins = bins[:-1]
    plt.subplot(2,2,2)
    plt.plot(bins , hists)
plt.show()
    

