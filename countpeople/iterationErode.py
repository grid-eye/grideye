import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
f_path = sys.argv[1]
frame_seq = sys.argv[2:]
img_set = np.load(f_path+"/imagedata.npy")
img_ave = np.load(f_path+"/avgtemp.npy")
binary_ones = np.ones((32,32))
kernel = np.ones((5,5))
def showImage(img):
    plt.imshow(img)
    plt.show()
for seq in frame_seq:
    img = img_set[seq]
    img = round(img)
    img=np.array(img,np.uint8)
    bool_res = img > (img_ave+2.0)
    binary = binary_ones * bool_res
    binary = np.array(binary,np.uint8)
    showImage(binary)
    while(1):
        erode = cv.erode(binary,kernel,iteration=1)
        img2 , contours , heirarchy = cv.findContours(erode,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        print("=====people count =====")
        print(len(contours))
        showImage(erode)
        binary = erode
        stop = input("continue?")
        if stop != "c":
            break
