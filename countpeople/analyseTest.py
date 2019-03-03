import numpy as np
import os
import sys
from extractBody import analyseFrameSequence
path = sys.argv[1]
d_num = []
if len(sys.argv) >= 3:
    arg_arr = sys.argv[2:]
    d_num = [int(item) for item in arg_arr]
else:
    quit()
all_frame = []
all_avgframe=[]
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
    analyseFrameSequence(frame_arr , frame_seq,avgtemp,False)
print()


