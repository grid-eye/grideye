import shutil as sh
import os
import sys
'''
    批量移动文件
'''
if len(sys.argv) < 3:
    raise ValueError("please specify a valid file path and dst path")
dirpath = sys.argv[1] #目录路径
dstpath = sys.argv[2]#目标路径
num =int( sys.argv[3]) #移动的目标数量
filename = "avgtemp.npy"
if len(sys.argv) > 4:
    if sys.argv[4]:
        filename = sys.argv[4]
for i in range(num) :
    if os.path.exists(dirpath+str(i)):
        sf = dirpath+str(i)+"/"+filename
        df = dstpath+str(i)
        sh.move(sf,df)
        print("move "+sf+" to "+df)

        



