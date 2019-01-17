import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import sys
from otsuBinarize import otsuThreshold as otsuThreshold
from interpolate import imageInterpolate
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    raise ValueError("please input the image's data's path and output dir")
if not os.path.exists(path):
    raise ValueError("please input a valid path")
def compatibleForCv(image,dtype = np.float32):
    image = np.array(image, dtype)
    return image
def calcOtsuThresh(diffdata,image_id,filter_process=False):
    plt.xticks([])
    plt.yticks([])
    #img = compatibleForCv(diffdata)
    img = np.array(diffdata,np.float32)
    print("img's dtype is :")
    print(img.dtype)
    median = cv.medianBlur(img,5)
    if median.max() > 2.5:
        print(median)
        print("median filter is %.2f"%(median.max()))
        print(median.dtype)
        gaussian = cv.GaussianBlur(img,(5,5),0)
        gaussian = np.round(gaussian,1)
        ret,thresh = otsuThreshold(gaussian,1024)
#        ret, otsu = cv.threshold(gaussian,-6.0,6.0,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return ret
    else:
        return None
allframe = np.load(path+"/imagedata.npy")
average = np.load(path+"/avgtemp.npy")
print("allframe's dtype is "+str(allframe.dtype))
print("average's dtype is "+str(average.dtype))
allframe = imageInterpolate(allframe )
average = imageInterpolate(average)
diffdata = []
#计算每一帧和当前温度的差值
for i in allframe:
    diffdata.append(i - average)
diffdata = np.array(diffdata)
print(diffdata.dtype)
diffdata = np.round(diffdata,2)
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


