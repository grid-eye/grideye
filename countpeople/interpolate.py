import numpy as np
import math
from scipy.interpolate import griddata
#因为项目保存的图像数据是8*8,所以分析时要对图像进行插值
row = 32
col = 32
points = [(math.floor(i/8),i%8) for i in range(64)]
grid_x,grid_y = np.mgrid[0:7:32j,0:7:32j]
def imageInterpolate(img ,method = 'linear'):
    '''
    默认内插方式是linear,也可以传递cubic等
    '''
    if len(img.shape) == 2:
        #只有一帧图片
        return griddata(points,img.ravel() ,(grid_x,grid_y),method=method)
    else:
        #多帧图片
        newImage = []
        for item in img:
            newImage.append(griddata(points,item.ravel(),(grid_x,grid_y),method=method))
        return np.array(newImage)



