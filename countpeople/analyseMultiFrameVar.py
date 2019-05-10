import sys
import os
import numpy as np
from analyseFrameVar import analyseSequence
#=============================================================================
#=用于确定方差的阈值，有人的帧和无人的帧的方差的比较,测试数据有足够大=
#=============================================================================
def saveMultiDirFrameVar(path , start,end,avgtemp,bg):
    path_arr = []
    ret_arr = []#保存返回结果
    all_var_arr,all_var_diff_arr = [],[]
    all_select_list = []
    for i in range(start,end):
        actual_dir = path+str(i)
        actual_path = actual_dir+"/imagedata.npy"
        if os.path.exists(actual_path) == False:
            continue
        frames=np.load(actual_path)
        if not bg:
            sel_path = actual_dir +"/human_data.npy"
            human_data = np.load(sel_path)
            select_list = human_data.tolist()
        else:
            select_list = [i for i in range(0,frames.shape[0])]
        show_frame = False
        ret = analyseSequence(frames,avgtemp,select_list,show_frame)
        all_var_arr += ret[0]
        all_var_diff_arr += ret[1]
    frame_sum = len(all_var_arr)

    print("=================frame sum is %d =================="%(frame_sum))
    print("=============max val is %.3f================="%(max(all_var_arr)))
    print("=============min val is %.3f==============="%(min(all_var_arr)))
    print("=============diff max val is %.3f================="%(max(all_var_diff_arr)))
    print("=============diff min val is %.3f==============="%(min(all_var_diff_arr)))
    postfix = path.split("/")[-1]
    outputDir ="analyseVar/"+ postfix
    np.save(outputDir+"var_diff_arr.npy",np.array(all_var_diff_arr))
    np.save(outputDir+"var_arr.npy",np.array(all_var_arr))
    print("sucessfully save in %sxx "%(outputDir))
def getDefaultPath():
    return [
                ("test/2019-3-12-second-",[1,5]),
                ("test/2019-3-26-",[1,4]),
                ("test/2019-3-31-",[1,2]),
                ("test/2019-3-31-high-",[1,4]),
                ("test/2019-3-19-first-",[1,2])
            ]

path = sys.argv[1]
if sys.argv[1] == "human":
    bg = False
else:
    bg = True
    start,end  = [int(t) for t in sys.argv[2:]]
all_frame  =[]
all_var_arr =[]
all_var_diff_arr=[]
if not bg:
    fg_path_arr = getDefaultPath()
    for path,ranges in fg_path_arr:
        start,end = ranges
        avgtemp = np.load(path+str(start)+"/avgtemp.npy")
        saveMultiDirFrameVar(path,start,end,avgtemp,False)
else:
    avgtemp = np.load(path+str(start)+"/avgtemp.npy")
    saveMultiDirFrameVar(path,start,end,avgtemp,True)
