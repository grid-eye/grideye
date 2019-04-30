import numpy as np
import sys
import random
path = sys.argv[1]
all_image = np.load(path)
gauss_mean = np.zeros(all_image[0].shape)
gauss_var = gauss_mean.copy()
sample_num = 80
gauss_sample = all_image[0:sample_num]
row,col = all_image[0].shape
high_temp = [2.5,2.7,3,3.5,3.6,2.6,2.1,2.4,2.6,2.8]
for i in range(row):
    for j in range(col):
        gauss_mean[i][j] = round(np.mean(gauss_sample[0:sample_num][i][j]),2)
        gauss_var[i][j] = round(np.var(gauss_sample[0:sample_num][i][j]),2)
frame_arr = []
k = 1500
for n in range(k):
    temp = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            temp[i][j] = np.round(random.gauss(gauss_mean[i][j],gauss_var[i][j]),2)
    frame_arr.append(temp)
print(frame_arr[0])
for n in range(300,k):
    temp = 0
    for 


