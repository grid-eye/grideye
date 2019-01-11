import os
import sys
from countpeople.countpeople import CountPeople as CP
from countpeople.cal_ave_bgtemperature import readBgTemperature 
"""
这个文件是自动收集n次m帧数据
"""
n ,m = 20,2000
if len(sys.argv) >3 :
        try:
            currDir = sys.argv[1]
            n = int(sys.argv[2])
            m = int(sys.argv[3])
        except:
            raise ValueError("please input a valid number ,default is 2000")
else:
    if len(sys.argv) > 2:
        try:
            currDir = sys.argv[1]
            n = int(sys.argv[2])
        except:
            raise ValueError("please input a valid number ,default is 2000")
    else:
        if len(sys.argv) > 1:
            currDir = sys.argv[1]
        else:
            raise ValueError("please specified a valid output dir for the imagedata")

cp = CP()
counter = 0
curr = os.path.abspath(os.path.dirname(__file__))
if curr.endswith("grideye"):
    curr += "/countpeople"
bgactual = curr+"/imagedata"+"/"+currDir
imageactual = curr+"/images"+"/"+currDir
cp.setPackageDir(curr)
cp.setBgDir("imagedata")
while counter < n:
    counter+=1
    print("the %dth whiles"%(counter))
    bgtempdir =bgactual+str(counter)
    cudr = currDir+str(counter)
    cp.setCustomDir(cudr)
    imagedir = imageactual+str(counter)
    readBgTemperature(400,bgtempdir)
    cp.process(m,imagedir)
print("sucessfully test all picture")
