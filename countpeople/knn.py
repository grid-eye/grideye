import numpy as np
import sys
import math


def calculateHighTemperature(img_ravel, split_index):
    partion = img_ravel[split_index]
    sub_array = img_ravel[np.where(img_ravel >= partion)]
    ave = np.average(sub_array)
    return ave


def calculateFeature(allframe, avgtemp, select_index, label):
    dataSet = []
    for i in select_index:
        curr_frame = allframe[i]
        diff_frame = curr_frame - avgtemp
        img_ravel = diff_frame.ravel()
        var = round(np.var(img_ravel), 2)
        '''
        ave = round(np.average(diff_frame), 2)
        split_index = math.ceil(img_ravel.size*5/6)
        high_temperature_ave = round(
            calculateHighTemperature(img_ravel, split_index), 2)
        sub_index_array = np.where(img_ravel > hist_x_thresh)
        if len(sub_index_array) == 0:
            hist_over_thresh = 0
        else:
            hist_over_thresh = sub_index_array[0].size
        '''
        sample = (i, var,  label)
        dataSet.append(sample)
    return np.array(dataSet)


def showSample(dataSet):
    for item in dataSet:
        print(item)
def createDataSet(path , label = 0):
    frame = np.load(path+"/imagedata.npy")
    avgtemp = np.load(bg_path+"/avgtemp.npy")
    if label == 1:
        human_seq = np.load(path+"/human_data.npy")
        half= int(human_frame.shape[0]/2)
        select_train = human_seq[0:half]
        select_test = human_seq[half:]
    else:
        half = int(frame.shape[0]/2)
        select_train =[i for i in range(0,half)] 
        select_test = [i for i in range(half,frame.shape[0])]
    trainDataSet = calculateFeature(frame,avgtemp,select_train,label)
    testDataSet =  calculateFeature(frame,avgtemp,select_test,label)
    return trainDataSet,testDataSet
def getOneKindSampleSet(path_arr,label):
    trainSet,testSet = [],[]
    for path in path_arr :
        train,test  = createDataSet(bg_path,label)
        trainSet.append(train)
        testSet.append(test)
    return np.array(trainSet),np.array(testSet)

def createMultiDirSampleSet(bg_path_arr,fg_path_arr):
    trainSet ,testSet = getOneKindSampleSet(bg_path_arr,0)
    fgSet = getOneKindSampleSet(fg_path_arr,1)
    trainSet = np.append(trainSet,fgSet[0],axis=0)
    testSet = np.append(testSet,fgSet[1],axis=0)
    return trainSet,testSet

def createSampleSet(bg_path,fg_paths):
    bgframe = np.load(bg_path+"/imagedata.npy")
    avgtemp_bg = np.load(bg_path+"/avgtemp.npy")
    select_index = [i for i in range(int(bgframe.shape[0]/5))]
    bgDataSet = calculateFeature(bgframe, avgtemp_bg, select_index, 0)
    # showSample(bgDataSet)
    fgDataSet =np.zeros((0,0))
    for i in range(len(fg_paths)):
        fg_path = fg_paths[i]
        fgframe = np.load(fg_path+"/imagedata.npy")
        avgtemp_fg = np.load(fg_path+"/avgtemp.npy")
        human_frame = np.load(fg_path+"/human_data.npy")
        half = int(human_frame.shape[0]/2)
        if human_frame.shape[0] == 0:
            continue
        fgData = calculateFeature(fgframe, avgtemp_fg, human_frame[0:half], 1)
        if fgDataSet.shape[0]>0:
            fgDataSet = np.append(fgDataSet,fgData,axis=0)
        else:
            fgDataSet = fgData
    fgDataSet = np.array(fgDataSet)
    return (bgDataSet,fgDataSet)

def getNormalDataSet(fgDataSet,bgDataSet):
    bglen = len(bgDataSet)
    allDataSet = bgDataSet + fgDataSet
    allDataSet = np.array(allDataSet)
    normalDataSet, ranges, minVals = normDataSet(allDataSet)
    allDataSet[:, 1] = normalDataSet
    bgDataSet, fgDataSet = allDataSet[0:bglen], allDataSet[bglen:]
    split_bg = int(len(bgDataSet)/2)
    split_fg = int(len(fgDataSet)/2)
    trainSet = np.append(bgDataSet[0:split_bg],fgDataSet[0:split_fg],axis=0)
    testSet =np.append( bgDataSet[split_bg:],fgDataSet[split_fg:],axis=0)
    return trainSet,testSet
    
def createTrainingSetAndTestSet(bg_path, fg_path):
    if type(bg_path) == str:
        bgDataSet,fgDataSet = createSampleSet(bg_path,fg_path)
    elif type(bg_path)==list:
        bgDataSet,fgDataSet =createMultiDirSampleSet(bg_path,fg_path) 
    return getNormalDataSet(bgDataSet,fgDataSet)


def normDataSet(dataSet):  # 归一化数据集
    '''
     Min-Max scaling
    '''
    minVals = dataSet[:, 1].min(0)
    maxVals = dataSet[:, 1].max(0)
    ranges = maxVals - minVals
    n = dataSet.shape[0]
    normalDataSet = dataSet[:, 1] - minVals
    normalDataSet = normalDataSet/ranges
    return normalDataSet, ranges, minVals


def knnClassify(trainingSet, labels, test, weight, k=5):
    diff_set = test[1] - trainingSet
    distance = abs(diff_set)
    sortedDistanceIndex = np.argsort(distance)
    classCount = {}
    for i in range(k):
        trainIndex = sortedDistanceIndex[i]
        voteLabel = labels[trainIndex]
        classCount[voteLabel] = classCount.get(voteLabel, 0)+1
    sortedClassCount = sorted(classCount.items(), key=lambda d: d[1],reverse=True)
    return sortedClassCount


def train(testSet, trainSet, weight, k):
    weight = np.tile(weight, (trainSet.shape[0], 1))
    errorCount = 0
    for i in range(testSet.shape[0]):
        actual_label = testSet[i][2]
        votedLabel ,votedCount= knnClassify(
                trainSet[:,1], trainSet[:, 2], testSet[i], weight, k)
        if votedLabel != actual_label:
            errorCount += 1
    return errorCount


def knnTrain(trainSet, testSet, k=5):
    '''
    trainSeq, trainSet,trainLabels = trainingSet[:,
        0],trainingSet[:,1:5],trainingSet[:,5]
    testSeq,testSet,testLabels = testSet[:,0],testSet[:,1:5],testSet[:,5]
    weight_array = np.linspace(0.1,1,10)
    weight_array = np.tile(weight_array,(4,1))
    min_count = 1000000
    min_tuple = None
    for f in weight_array[0][4:]:
        for j in weight_array[1]:
            for n in weight_array[2]:
                for m in weight_array[3]:
                    weight=(f,j,n,m)
                    errorCount = train(testSet,trainSet,weight,k)
                    if errorCount < min_count:
                        min_count = errorCount
                        min_tuple = weight
   '''
    weight = (1,)
    errorCount = train(testSet,trainSet,weight,k)
    dataSize = testSet.shape[0]
    print("================errorCount is %d================"%(errorCount))
    print("===============error accuracy is %.5f==========="%(errorCount/dataSize))
    print("===============correct accuracy is %.5f==========="%((dataSize-errorCount)/dataSize))
def getDefaultBgpathAndFgpath():
    pass    

if __name__ == "__main__":
    if len(sys.argv) > 2:
        bg_path = sys.argv[1]
        fg_path = sys.argv[2]
    else:
        bg_path ,fg_path = getDefaultBgpathAndFgpath()

    trainSet,testSet = createTrainingSetAndTestSet(bg_path,fg_path)
    ws = [3,5,7,9 ,11,13,15,17,19]
    for i in ws:
        print("===================%d=================="%(i))
        knnTrain(trainSet,testSet,i)
    '''
    knnTrain(trainSet,testSet,9)
    '''

