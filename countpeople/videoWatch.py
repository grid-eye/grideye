import numpy as np
import cv2 as cv
import numpy as np
import sys
path = sys.argv[1]
all_frame = np.load(path)
cv.namedWindow("image",cv.WINDOW_NORMAL)
temp = np.zeros((8,8),np.uint8)
bg_frame = all_frame[0:40]
avgtemp = np.average(bg_frame,axis= 0)
for i in range(40,all_frame.shape[0]):
    print("=============the %dth frame================"%(i))
    diff = all_frame[i] - avgtemp
    temp[np.where(diff > 1)] = 255
    paint_img = cv.resize (temp,(16,16),interpolation= cv.INTER_CUBIC)
    cv.imshow("image",paint_img)
    cv.waitKey(10)
    temp.fill(0)
cv.destroyAllWindows()
    
