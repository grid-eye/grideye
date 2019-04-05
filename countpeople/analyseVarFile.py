import numpy as np
import sys
def analyseVar(var_arr,interval=[0.10 ,0.125,0.15,0.175,0.2,0.225,  0.25]):
    #interval =[0.4,0.425,0.45, 0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7]
    print("===================maximun of var_arr is =================")
    print(var_arr.max())
    print("====================minimum of var_arr is================")
    print(var_arr.min())
    for i in interval:
        print("================%.4f================="%(i))
        sub_index = np.where(var_arr >= i)[0]
        print("=============var >= %.4f=============="%(i))
        print(sub_index.size)
        print("==========%.3f%%================"%(sub_index.size/var_arr.size*100))
        sub_index = np.where(var_arr <i)[0]
        print("=============var < %.4f=============="%(i))
        print(sub_index.size)
        print("==========%.3f%%================"%(sub_index.size/var_arr.size*100))
if __name__ == "__main__":
    path = sys.argv[1]
    var_arr = np.load(path)
    print("================%s shape is============"%(path))
    print(var_arr.shape)
    max_var = var_arr.max()
    min_var = var_arr.min()
    interval = [0.10 ,0.125,0.15,0.175,0.2,0.225,  0.25]
    analyseVar(var_arr,interval)



