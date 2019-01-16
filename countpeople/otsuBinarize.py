import numpy as np
def findThresh(histogram , total,ranges = (-6,6),interval =
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
        if between >= max:
            threshold1 = i
            if between > max:
                threshold2 = i 
            max = between
    return (threshold1+threshold2)/2.0
def calcHistogram(images):
    hist ,bins = np.histogram(img.ravel() ,[0] , bins =120 , range=(-6,6))
    bins = bins[:-1]
    freqMap  = dict.fromkeys(bins ,0)
    for i in range(hist.shape[0]):
        freqMap[bins[i]] = hist[i]
    return freqMap
def OtsuThreshold(images , total,ranges = (-6,6),interval =0.1):
    histogram = calcHistogram(images)
    ret = findThresh(histogram,total,ranges,interval)
    print("ret is %.1f"%(ret))
    shape = images.shape
    hist  = np.ones(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if images[i][j] < ret:
                images[i][j] = 0
    return (ret,hist)
