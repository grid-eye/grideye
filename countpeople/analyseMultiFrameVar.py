import sys
import os
import numpy as np
from analyseFrameVar import analyseSequence
#=============================================================================
#=用于确定方差的阈值，有人的帧和无人的帧的方差的比较,测试数据有足够大=
#=============================================================================
path = sys.argv[1]
if sys.argv[2] == "human":
    bg = False
    dir_arr = sys.argv[3:]
else:
    bg = True
    dir_arr = sys.argv[2:]
all_frame  =[]
all_ave = []
path_arr =[]
all_var_arr =[]
all_var_diff_arr=[]
all_select_list = []
for seq in dir_arr:
    actual_dir = path+seq
    actual_path = actual_dir+"/imagedata.npy"
    frames=np.load(actual_path)
    if not bg:
        sel_path = actual_dir +"/human_data.npy"
        human_data = np.load(sel_path)
        select_list = human_data.tolist()
    else:
        select_list = [i for i in range(0,frames.shape[0])]
    all_select_list.append(select_list)
    path_arr.append(actual_path)
    actual_avg_path=actual_dir+"/avgtemp.npy"
    avgframe = np.load(actual_avg_path)
    all_ave.append(avgframe)
    all_frame.append(frames)
ret_arr = []#保存返回结果
for i in range(len(all_select_list )):
    curr_path = path_arr[i]
    print("analyse "+curr_path)
    frame_seq = all_frame[i]
    avgtemp = all_ave[i]
    argarray = all_select_list[i]
    show_frame = False
    ret = analyseSequence(frame_seq,avgtemp,argarray,show_frame)
    all_var_arr += ret[0]
    all_var_diff_arr += ret[1]
frame_sum = len(all_var_arr)
split_res = path.split("/")
outputDir ="analyseVar/"+ split_res[-1]
np.save(outputDir+"var_arr.npy",np.array(all_var_arr))
np.save(outputDir+"var_diff_arr.npy",np.array(all_var_diff_arr))
print("=================frame sum is %d =================="%(frame_sum))
print("=============max val is %.3f================="%(max(all_var_arr)))
print("=============min val is %.3f==============="%(min(all_var_arr)))
print("=============diff max val is %.3f================="%(max(all_var_diff_arr)))
print("=============diff min val is %.3f==============="%(min(all_var_diff_arr)))
