import sys
import os
import numpy as np
from analyseFrameAverageThresh import analyseSequence
#=============================================================================
#=用于确定平均温度的阈值，有人的帧和无人的帧的平均温度的比较,测试数据有足够大=
#=============================================================================
path = sys.argv[1]
dir_arr = sys.argv[2:]
all_frame  =[]
all_ave = []
path_arr =[]
for seq in dir_arr:
    actual_dir = path+seq
    actual_path = actual_dir+"/imagedata.npy"
    path_arr.append(actual_path)
    actual_avg_path=actual_dir+"/avgtemp.npy"
    frames=np.load(actual_path)
    avgframe = np.load(actual_avg_path)
    all_ave.append(avgframe)
    all_frame.append(frames)
ret_arr = []#保存返回结果
for i in range(len(all_frame)):
    curr_path = path_arr[i]
    print("analyse "+curr_path)
    frame_seq = all_frame[i]
    avgtemp = all_ave[i]
    argarray = [i for i in range(len(all_frame[i]))]
    show_frame = False
    ret = analyseSequence(frame_seq,avgtemp,argarray,show_frame)
    ret_arr.append(ret)
frame_sum = 0
over_thresh_sum= 0
for fsum,over_thresh in ret_arr:
    frame_sum += fsum
    over_thresh_sum += over_thresh
print("=================frame sum is %d =================="%(frame_sum))
print("=================over thresh_sum is %d=============="%(over_thresh_sum))
print("==================frame with man accuracy is %.2f====================="%(over_thresh_sum/frame_sum))
print("===================frae without human accuracy is %.2f================="%((frame_sum-over_thresh_sum)/frame_sum))
