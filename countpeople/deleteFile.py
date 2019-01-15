import os
import sys
'''
    批量移动文件
'''
if len(sys.argv) < 3:
    raise ValueError("please specify a valid file path and dst path")
dirpath = sys.argv[1] #目录路径
num =int( sys.argv[2]) #删除的目标数量
filename = "diffdata.npy"
if len(sys.argv) > 3:
    if sys.argv[3]:
        filename = sys.argv[3]
for i in range(num) :
    if os.path.exists(dirpath+str(i)):
        sf = dirpath+str(i)+"/"+filename
        try:
            os.remove(sf)
            print("delete "+sf+" sucessfully")
        except FileNotFoundError:
            print("no such file")



