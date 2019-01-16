import numpy as np
def otsuThreshold(histogram , total,ranges = (-6,6),interval =
        0.1,dtype=np.float32):
    '''
    arg:
        histogram:图像的直方图分布(数据类型是float32)
        total:图像的pixel个数
    '''
    sum = 0 
    sample = np.arange(ranges[0],ranges[1],0.1,dtype)
    for i in sample:
        sum += i*histogram[i]
    sumB = 0
    wB = 0
    wF = 0
    mB = 0
    mF = 0
    max = 0.0
    between = 0
    threshold1,threshold2 = .0,.0
    for  i in sample:
        wB+=historam[i]
        if wB == 0 :
            continue
        wF = total - wB
        if wF == 0
            break
        sumB += i*histogram[i]
        mB = sumB/wB
        mF = (sum - sumB) /WF
        between = wB*wF*(mB-mF)*(mB-mF)
        if bwib
