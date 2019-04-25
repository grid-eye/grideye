import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import sys
from interpolate import imageInterpolate

if len(sys.argv) > 2:
    path = sys.argv[1]
    output = sys.argv[2]
else:
    raise ValueError("please input the image's data's path and output dir")
if not os.path.exists(path):
    raise ValueError("please input a valid path")

def compatibleForCv(image):
    return image.astype(np.uint8)
def convertData2Image(imageData,image_id,filter_process=False):
    if filter_process == True:
        plt.subplot(221)
    else:
        plt.subplot(211)
    imageData = imageInterpolate(imageData,"cubic")
    plt.imshow(imageData)
    plt.xticks([])
    plt.yticks([])
    plt.title("image_%d"%(image_id))
    if filter_process == True:
        img = compatibleForCv(imageData)
        median = cv.medianBlur(img,5)
        plt.subplot(222)
        plt.imshow(median)
        plt.title("median_%d"%(image_id))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(223)
        gaussian = cv.GaussianBlur(img,(5,5),0)
        plt.imshow(gaussian)
        plt.title("gauss_%d"%(image_id))
        plt.xticks([])
        plt.yticks([])
        ret, otsu = cv.threshold(gaussian,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        plt.subplot(224)
        plt.imshow(otsu)
        plt.title("gauss_otsu_%d"%(image_id))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig("%s/image_%d"%(output,image_id))
    plt.clf()
imagedata = np.load(path)
print("load data sucessfully!")
if len(sys.argv) > 3:
    sel_frame = [ int(s) for s in sys.argv[3:]]
else:
    sel_frame = [i for i  in range(imagedata.shape[0])]
print(imagedata.shape)
for i in sel_frame:
    print('%dth frames pic'%(i))
    convertData2Image(imagedata[i],i,False)
print("save sucessfully")

