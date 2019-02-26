import cv2 as cv
import numpy as np
import sys
f_path = sys.argv[1]
frame_seq = sys.argv[2:]
img_set = np.load(f_path+"/imagedata.npy")
img_ave = np.load(f_path+"/average.npy")
binary_ones = np.ones((32,32))
for seq in frame_seq:
    img = img_set[seq]
    img = round(img)
    img=np.array(img,np.uint8)
    bool_res = img > (img_ave+2.0)
    binary = binary_ones * bool_res
    while(1):
        
    
