import cv2 as cv
import numpy as np
import sys
path = sys.argv[1]
frames = np.load(path+"/imagedata.npy")
avgtemp = np.load(path+"/avgtemp.npy")
if len(sys.argv)>2 :
    sel_frames= [ int(item) for item in sys.argv[2:]]
else:
    sel_frames = [i for i in range(frames.shape[0])]
window_name = "image"
print("name is "+window_name)
cv.namedWindow(window_name,cv.WINDOW_NORMAL)
for item in sel_frames:
    curr_frame = frames[item]
    diff_temp = curr_frame - avgtemp
    diff_temp = np.round(diff_temp)
    diff_temp = diff_temp.astype(np.uint8)
    diff_temp[np.where(diff_temp < 1.5)] = 0
    diff_temp[np.where(diff_temp>=1.5)] = 255
    print("imshow")
    cv.imshow(window_name,diff_temp)
    print("waitKey")
    cv.waitKey(20)
cv.destroyAllWindows()
