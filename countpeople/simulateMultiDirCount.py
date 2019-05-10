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
error_frame_seq = []
all_area_ret = []
people_sum = 0 
for postfix in arg_arr:
    subpath = path+postfix+"/"+"imagedata.npy"
    avgpath = path+postfix+"/"+"avgtemp.npy"
    frame_seq = np.load(subpath)
    avgtemp = np.load(avgpath)
    frame_arr = [i for i in range(frame_seq.shape[0])]
    print("===============analyse a frame seqeunce in %s =================="%(subpath))
    area_ret,people_num = analyseFrameSequence(frame_arr , frame_seq,avgtemp,path+postfix,False)
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



