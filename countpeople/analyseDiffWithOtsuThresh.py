import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    raise ValueError("please input the image's data's path and output dir")
if not os.path.exists(path):
    raise ValueError("please input a valid path")
def compatibleForCv(image):
    cv.imwrite("temp.png",image)
    return cv.imread("temp.png",0)
def calcOtsuThresh(diffdata,image_id,filter_process=False):
    plt.xticks([])
    plt.yticks([])
    img = compatibleForCv(diffdata)
    median = cv.medianBlur(img,5)
    if median.max() > 2.5:
        print(median)
        print("median filter is %.2f"%(median.max()))
        gaussian = cv.GaussianBlur(img,(5,5),0)
        ret, otsu = cv.threshold(gaussian,-6.0,6.0,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return ret
    else:
        return None
allframe = np.load(path+"/diffdata.npy")
average = np.load(path+"/avgtemp.npy")
diffdata = []
#计算每一帧和当前温度的差值
for i in allframe:
    diffdata.append(allframe[i] - average)
diffdata = np.array(diffdata)
diffdata = np.round(diffdata,1)
print("load data sucessfully!")
print(diffdata.shape)
result =[]
for i in range(len(diffdata)):
    #print('%dth frames pic'%(i))
    if diffdata[i].max() < 2.6:
        continue
    ret = calcOtsuThresh(diffdata[i],i,True)
    if ret :
        result.append(ret)
print("sucessfully process all frame")
result = np.array(result)
print(result)
print("the max thresh of the result is %.2f"%(result.max()))
print("the minimum thresh of the result is %.2f"%(result.min()))

