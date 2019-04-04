import numpy as np
import sys
from analyseVarFile import analyseVar
prefix = "analyseVar"
files = ["2019-3-20-var_diff_arr.npy","2019-3-31-high-var_diff_arr.npy","2019-3-12-second-var_diff_arr.npy","2019-3-26-var_diff_arr.npy"]
file_path =[ prefix+"/"+item for item in files]
all_var = np.array([])
for f in file_path:
    var_arr = np.load(f)
    all_var = np.append(all_var,var_arr)
print("length of all var_arr is %d "%(all_var.shape[0]))
analyseVar(all_var)

