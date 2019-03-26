import numpy as np
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
    #img = compatibleForCv(diffdata)
    img = np.array(diffdata,np.float32)
    median = cv.medianBlur(img,5)
    if median.max() > 2.5:
        gaussian = cv.GaussianBlur(img,(5,5),0)
        gaussian = np.round(gaussian,1)
        ret,thresh = otsuThreshold(gaussian,1024)
#        ret, otsu = cv.threshold(gaussian,-6.0,6.0,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return ret
    else:
        return None
allframe = np.load(path+"/imagedata.npy")
average = np.load(path+"/avgtemp.npy")

if len(sys.argv) >2 :
    selframe = [int(i) for i in sys.argv[2:]]
else:
    selframe = [i for i in range(allframe.shape[0])]
print("allframe's dtype is "+str(allframe.dtype))
print("average's dtype is "+str(average.dtype))
#allframe = imageInterpolate(selframe )
#average = imageInterpolate(average)
diffdata = []
#计算每一帧和当前温度的差值
for i in selframe:
    diffdata.append(allframe[i] - average)
diffdata = np.array(diffdata)
print(diffdata.dtype)
diffdata = np.round(diffdata,2)
print("load data sucessfully!")
print(diffdata.shape)
result =[]
for i in range(len(diffdata)):
    print('%dth frames pic'%(selframe[i]))
    print(allframe[selframe[i]])
    print("diff frame is")
    print(diffdata)
    if diffdata[i].max() < 2.:
        continue
    ret = calcOtsuThresh(diffdata[i],i,True)
    if ret :
        result.append(ret)
print("sucessfully process all frame")
result = np.array(result)
print(result)
if result:
    print("the max thresh of the result is %.2f"%(result.max()))
    print("the minimum thresh of the result is %.2f"%(result.min()))


