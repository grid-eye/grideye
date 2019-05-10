import sys
import os
import numpy as np
from analyseFrameAverageThresh import analyseSequence
#=============================================================================
#=用于确定平均温度的阈值，有人的帧和无人的帧的平均温度的比较,测试数据有足够大=
#=============================================================================
def getDefaultFgPath():
    return [
             ("images/2019-2-2-first",[1,5]),
             ("test/2019-3-12-second-",[1,5]),
             ("test/2019-3-19-first-",[1,2]),
             ("test/2019-3-26-",[1,4]),
             ("test/2019-3-31-high-",[1,4]),
             ("test/2019-3-31-",[1,2])
            ]

def showThreshMap(thresh_map):
    for k,v in thresh_map:
        frame_sum = v[0]
        over_thresh_sum = v[1]
        print("thresh is %.3f "%(k))
        print("=================frame sum is %d =================="%(frame_sum))
        print("=================over thresh_sum is %d=============="%(over_thresh_sum))
        print("==================frame with man accuracy is %.2f====================="%(over_thresh_sum/frame_sum))
        print("=================fg sum is %d==========="%(frame_sum - over_thresh_sum))
        print("===================frame without human accuracy is %.2f================="%((frame_sum-over_thresh_sum)/frame_sum))
def analyseMultiDirFrame(path,start,end,thresh_arr,bg):
    avgtemp = np.load(path+str(start)+"/avgtemp.npy")
    ret_arr = []#保存返回结果
    all_frame = []
    show_frame = False
    for seq in range(start,end):
        actual_dir = path+str(seq)
        actual_path = actual_dir+"/imagedata.npy"
        frame = np.load(actual_path)
        if os.path.exists(actual_path) == False:
            continue
        if not  bg:
            human_data = np.load(actual_dir+"/human_data.npy")
            temp = []
            for item in human_data:
                temp.append(frame[item])
            all_frame += temp
        else:
            all_frame += frame.tolist()
    all_frame = np.array(all_frame)
    print(all_frame.shape)
    thresh_map =[]
    seqarray = [i for i in range(all_frame.shape[0])]
    for item in thresh_arr:
        ret = analyseSequence(all_frame,avgtemp,seqarray , show_frame,thresh=item)
        thresh_map.append((item,ret))
    showThreshMap(thresh_map)
    return thresh_map
path = sys.argv[1]
if sys.argv[1] == "human":
    bg=False
else:
    s = 2
    bg=True
thresh_arr = [0.2, 0.25,0.3,0.35,0.4]
if bg:
    start ,end  = [int(s) for s in sys.argv[s:s+2]]
    result_map = analyseMultiDirFrame(path,start,end,thresh_arr)
else:
    fgPaths = getDefaultFgPath()
    final_thresh_map = {}
    for path,ranges in fgPaths:
        ret = analyseMultiDirFrame(path,ranges[0],ranges[1],thresh_arr,bg)
        if len(final_thresh_map) == 0:
            final_thresh_map=dict(ret)
        else:
            for k,num_arr in ret:
                v = final_thresh_map[k]
                final_thresh_map[k] = [v[0]+num_arr[0],v[1]+num_arr[1]]
    result_map  = sorted(final_thresh_map.items(),key=lambda d:d[1])
showThreshMap(result_map)
