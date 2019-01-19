import numpy as np
def findThresh(histogram , total,ranges = (-6,6),interval =
        0.1,dtype=np.float32):
    '''
    arg:
        histogram:图像的直方图分布(数据类型是float32)
        total:图像的pixel个数
    '''
    sum = 0 
    print("the histogram is ")
    print(histogram)
    allsum =  0
    for k,v in histogram.items():
        allsum += v
    print("allsum = %d"%(allsum))
    allsum = 0
    sortHist = sorted(histogram.items(),key = lambda item:item[0])
    for k,v in sortHist:
        sum += k*v
        allsum  += v
    print("allsum = %d"%(allsum))
    sumB = 0
    wB = 0
    wF = 0
    mB = 0
    mF = 0
    max = 0.0
    between = 0
    threshold1,threshold2 = .0,.0
    for  i,v in sortHist:
        wB+=v
        if wB  ==  0 :
            continue
        wF = total - wB
        if wF == 0 :
            break
        sumB += i*v
        mB = sumB/wB
        mF = (sum - sumB) /wF
        between = wB*wF*(mB-mF)*(mB-mF)
        if between >= max:
            threshold1 = i
            if between > max:
                threshold2 = i 
            max = between
    return (threshold1+threshold2)/2.0
def calcHistogram(images,ranges = (-6,6) ):
    bins = (ranges[1] - ranges[0]) * 10
    images = np.round(images,1)
    hist ,bins = np.histogram(images.ravel() , bins=bins , range=(-6,6))
    print(hist.sum())
    print(bins)
    bins = bins[:-1]
    freqMap  = {}
    for i in range(hist.shape[0]):
        freqMap[round(bins[i],1)] = hist[i]
    print(freqMap)
    return freqMap
def otsuThreshold(images , total,ranges = (-6,6),interval =0.1,thre = None):
    histogram = calcHistogram(images)
    if thre:
        ret = thre
    else:
        ret = findThresh(histogram,total,ranges,interval)
    print("ret is %.1f"%(ret))
    shape = images.shape
    hist  = np.ones(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if images[i][j] < ret:
                hist[i][j] = 0
    return (ret,hist)
