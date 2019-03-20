import numpy as np
import os
import sys
from simulateCount import analyseFrameSequence
path = sys.argv[1]
d_num = []
if len(sys.argv) >= 3:
    arg_arr = sys.argv[2:]
    d_num = [int(item) for item in arg_arr]
else:
    quit()
all_frame = []
all_avgframe=[]
all_area_ret = []
error_frame_seq = []
people_sum = 0 
for postfix in arg_arr:
    subpath = path+postfix+"/"+"imagedata.npy"
    avgpath = path+postfix+"/"+"avgtemp.npy"
    all_frame.append(np.load(subpath))
    all_avgframe.append(np.load(avgpath))
for i in range(len(all_frame)):
    frame_seq = all_frame[i]
    avgtemp = all_avgframe[i]
    frame_arr = [i for i in range(len(frame_seq))]
    print(frame_arr)
    print(len(frame_seq))
    print("===============analyse a frame seqeunce in %s =================="%(subpath))
    area_ret,people_num = analyseFrameSequence(frame_arr , frame_seq,avgtemp,False)
    if people_num == 0:
        error_frame_seq.append(d_num[i])
    people_sum += people_num
    all_area_ret.append(area_ret)
print()
print("=====all_area_ret is =================")
print(np.array(all_area_ret))
max_ret = []
for item in all_area_ret:
    max_ret.append(max(item))
print("{====max arr is ====")
print(max_ret)
print("====the maximum value is =====")
print(max(max_ret))
print("============people num is ===================")
print(people_sum)
print("============error frame seq is ==================")
print(error_frame_seq)



