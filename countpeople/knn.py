import numpy as np
import os
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
    avgtemp = np.load(path+"/avgtemp.npy")
    bg_proportion = 3/4
    fg_proportion = 4/5
    if label == 1:
        human_seq = np.load(path+"/human_data.npy")
        part= int(human_seq.shape[0]*fg_proportion)
        select_train = human_seq[0:part]
        select_test = human_seq[part:]
    else:
        part = int(frame.shape[0]*bg_proportion)
        select_train =[i for i in range(0,part)] 
        select_test = [i for i in range(part,frame.shape[0])]
    testDataSet =  calculateFeature(frame,avgtemp,select_test,label)
    trainDataSet =  calculateFeature(frame,avgtemp,select_train,label)
    return trainDataSet,testDataSet
def getOneKindSampleSet(path_arr,label):
    assign = False
    for path,start,end in path_arr :
        for i in range(start,end):
            real_path = path+str(i)
            if not os.path.exists(real_path):
                print(real_path)
                print("current path not existing")
                continue
            train,test  = createDataSet(real_path,label)
            if not assign:
                trainSet ,testSet = train,test
                assign = True
            else:
                trainSet = np.append(trainSet,train,axis=0)
                testSet = np.append(testSet,test,axis=0)
    return trainSet,testSet

def createMultiDirSampleSet(bg_path_arr,fg_path_arr):
    trainSet ,testSet = getOneKindSampleSet(bg_path_arr,0)
    print("bg set 'size is  %d"%(trainSet.shape[0] + testSet.shape[0]))
    fgTrainSet,fgTestSet = getOneKindSampleSet(fg_path_arr,1)
    print("fg set 'size is %d "%(fgTrainSet.shape[0] + fgTestSet.shape[0]))
    trainSet = np.append(trainSet,fgTrainSet,axis=0)
    testSet = np.append(testSet,fgTestSet,axis=0)
    print("trainSet's size is %d"%(trainSet.shape[0]))
    print("testSet's size is %d"%(testSet.shape[0]))
    return trainSet,testSet

def createSampleSet(bg_path,fg_paths):
    bgframe = np.load(bg_path+"/imagedata.npy")
    avgtemp_bg = np.load(bg_path+"/avgtemp.npy")
    select_index = [i for i in range(bgfame.shape[0])]
    bgDataSet = calculateFeature(bgframe, avgtemp_bg, select_index, 0)
    # showSample(bgDataSet)
    assign = False
    for i in range(len(fg_paths)):
        fg_path = fg_paths[i]
        fgframe = np.load(fg_path+"/imagedata.npy")
        avgtemp_fg = np.load(fg_path+"/avgtemp.npy")
        human_frame = np.load(fg_path+"/human_data.npy")
        part = int(human_frame.shape[0]/2)
        if human_frame.shape[0] == 0:
            continue
        fgData = calculateFeature(fgframe, avgtemp_fg, human_frame[0:part], 1)
        if assign:
            fgDataSet = np.append(fgDataSet,fgData,axis=0)
        else:
            fgDataSet = fgData
            assign = True
        part = int(bgDataSet.shape[0]*3/4)
        fg_part = int(fgDataSet.shape[0]*3/4)
        trainSet = np.append(bgDataSet[0:part],fgDataSet[0:fg_part],axis=0)
        testSet = np.append(bgDataSet[part:],fgDataSet[fg_part:],axis=0)
    return trainSet,testSet
def getNormalDataSet(trainSet,testSet):
    trainlen = len(trainSet)
    allDataSet = np.append(trainSet , testSet,axis=0)
    normalDataSet, ranges, minVals = normDataSet(allDataSet)
    allDataSet[:, 1] = normalDataSet
    trainSet,testSet = allDataSet[0:trainlen], allDataSet[trainlen:]
    return trainSet,testSet
def createTrainingSetAndTestSet(bg_path, fg_path):#创建测试数据集并进行归一化
    if type(bg_path) == str:
        print("case 1")
        trainSet,testSet = createSampleSet(bg_path,fg_path)
    elif type(bg_path)==list:
        print("case 2")
        trainSet,testSet =createMultiDirSampleSet(bg_path,fg_path) 
    print("trainSet is ")
    print(trainSet.shape)
    print("testSet is")
    print(testSet)
    return getNormalDataSet(trainSet,testSet)


def createTrainingSet(bg_path,fg_path):
    trainSet,testSet = createTrainingSetAndTestSet(bg_path,fg_path)
    return np.append(trainSet ,testSet,axis=0)
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
    classCount = {0:0,1:0}
    for i in range(k):
        trainIndex = sortedDistanceIndex[i]
        voteLabel = int(labels[trainIndex])
        classCount[voteLabel] = classCount.get(voteLabel, 0)+1
    sortedLabel = sorted(classCount.items() ,key=  lambda d:d[1],reverse = True)
    return sortedLabel[0]

def train(trainSet, testSet, weight, k):
    print("trainSet shape is ")
    print(trainSet.shape)
    print("testSet shape is ")
    print(testSet.shape)
    weight = np.tile(weight, (trainSet.shape[0], 1))
    errorCount = 0
    fg_count ,bg_count = 0, 0 
    fg_count_error ,bg_count_error = 0, 0
    for i in range(testSet.shape[0]):
        actual_label = testSet[i][2]
        votedLabel ,votedCount= knnClassify(
                trainSet[:,1], trainSet[:, 2], testSet[i], weight, k)
        if votedLabel != actual_label:
            errorCount += 1
        if actual_label == 0:
            bg_count == 0
            if actual_label != votedLabel:
                bg_count_error += 1
        else:
            fg_count = 0
            if actual_label != votedLabel:
                fg_count_error += 1
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
    errorCount = train(trainSet,testSet,weight,k)
    dataSize = testSet.shape[0]
    print("================errorCount is %d================"%(errorCount))
    print("===============error accuracy is %.5f==========="%(errorCount/dataSize))
    print("===============correct accuracy is %.5f==========="%((dataSize-errorCount)/dataSize))
def getDeafultTestSet():
    bg_paths = [

            ]
    fg_paths = [


            ]
    return bg_paths,fg_paths
def getDefaultBgpathAndFgpath():
    bg_paths =[
                ("images/2019-01-17-bgfirst",1,10)
            ]
    fg_paths=[
                ("test/2019-3-12-second-",1,5),
                ("test/2019-3-19-",1,2),
                ("test/2019-3-26-",1,4),
                ("test/2019-3-31-",1,2),
                ("test/2019-3-31-high-",1,4),
                ("test/2019-4-1-",1,3),
                ("images/2019-2-2-first",1,6)
            ]
    return bg_paths,fg_paths

if __name__ == "__main__":
    if len(sys.argv) > 2:
        bg_path = sys.argv[1]
        fg_path = sys.argv[2]
    else:
        bg_path ,fg_path = getDefaultBgpathAndFgpath()
    trainSet,testSet = createTrainingSetAndTestSet(bg_path,fg_path)
    print("trainSet.size is ")
    print(trainSet.shape)
    print("testSet.size is ")
    print(testSet.shape)
    ws = [3,5,7,9 ,11,13,15,17,19]
    for i in ws:
        print("===================%d=================="%(i))
        knnTrain(trainSet,testSet,i)
    '''
    knnTrain(trainSet,testSet,9)
    '''

