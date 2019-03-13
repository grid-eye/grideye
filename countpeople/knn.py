import numpy as np
import os
import sys
import math
def calculateHighTemperature(img_ravel,split_index):
    partion = img_ravel[split_index]
    sub_array = img_ravel[np.where(img_ravel >= partion)]
    ave = np.average(sub_array)
    return ave
def calculateFeature(allframe , avgtemp,select_index,label):
    dataSet =[]
    hist_x_thresh =2
    for i in select_index:
        curr_frame = allframe[i]
        diff_frame = curr_frame - avgtemp
        img_ravel = diff_frame.ravel()
        var  =round( np.var(img_ravel),2)
        ave =round( np.average(diff_frame),2)
        split_index = math.ceil(img_ravel.size*5/6)
        high_temperature_ave =round( calculateHighTemperature(img_ravel,split_index),2)
        sub_index_array = np.where(img_ravel > hist_x_thresh)
        if len(sub_index_array) == 0:
            hist_over_thresh = 0
        else:
            hist_over_thresh = sub_index_array[0].size
        sample = (i,var,ave,high_temperature_ave,hist_over_thresh,label)
        dataSet.append(sample) 
    return dataSet
def showSample(dataSet):
    for item in dataSet :
        print(item)
def createTrainingSetAndTestSet(bg_path,fg_path):
    bgframe = np.load(bg_path+"/imagedata.npy")
    avgtemp_bg = np.load(bg_path+"/avgtemp.npy")
    fgframe = np.load(fg_path+"/imagedata.npy")
    avgtemp_fg = np.load(fg_path+"/avgtemp.npy")
    human_frame = np.load(fg_path+"/human_data.npy")
    select_index = [i  for i in range(int(bgframe.shape[0]/5))]
    bgDataSet = calculateFeature(bgframe,avgtemp_bg,select_index,0)
    #showSample(bgDataSet)
    fgDataSet = calculateFeature(fgframe,avgtemp_fg,human_frame,1)
    #showSample(fgDataSet)
    bglen = len(bgDataSet)
    allDataSet = bgDataSet + fgDataSet
    print(type(allDataSet))
    allDataSet = np.array(allDataSet)
    print(allDataSet.shape)
    print(allDataSet.shape)
    normalDataSet,ranges,minVals = normDataSet(allDataSet)
    allDataSet[:,1:5] = normalDataSet
    print(type(allDataSet))
    bgDataSet ,fgDataSet = allDataSet[0:bglen],allDataSet[bglen:]
    split_bg = int(len(bgDataSet)/2)
    split_fg = int(len(fgDataSet)/2)
    
    trainSet = bgDataSet[0:split_bg].tolist()+fgDataSet[0:split_fg].tolist()
    print("==============train set is===================")
    #showSample(trainSet)
    testSet = bgDataSet[split_bg:].tolist()+fgDataSet[split_fg:].tolist()
    print("==============test set is===================")
    #showSample(testSet)
    return np.array(trainSet),np.array(testSet)
def normDataSet(dataSet):#归一化数据集
    '''
     Min-Max scaling
    '''
    minVals = dataSet[:,1:5].min(0)
    maxVals = dataSet[:,1:5].max(0)
    ranges = maxVals - minVals
    n = dataSet.shape[0]
    normalDataSet = dataSet[:,1:5] - np.tile(minVals,(n,1))
    normalDataSet = normalDataSet/np.tile(ranges,(n,1))
    return normalDataSet,ranges,minVals
def knnClassify(trainingSet,labels , test,k=5):
    m = trainingSet.shape[0]
    testExtension = np.tile(test[1:5],(m,1))
    diff_set = testExtension - trainingSet[:,1:5]
    weight = np.array([1,0.3,0.1,1])
    weight = np.tile(weight,(m,1))
    diff_set = diff_set * weight
    diff_set = diff_set**2
    sqDistance = diff_set.sum(axis=1)
    distance = sqDistance**(0.5)
    sortedDistanceIndex = np.argsort(distance)
    classCount={}
    for i in range(k):
        trainIndex = sortedDistanceIndex[i]
        voteLabel = labels[trainIndex]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.items(),key = lambda d:d[1])
    return sortedClassCount[0][0]
def knnTrain(trainSet,testSet,k=5):
    '''
    trainSeq, trainSet,trainLabels = trainingSet[:,0],trainingSet[:,1:5],trainingSet[:,5]
    testSeq,testSet,testLabels = testSet[:,0],testSet[:,1:5],testSet[:,5]
    '''
    errorCount = 0
    for i in range(testSet.shape[0]):
        actual_label =testSet[i][5]
        votedLabel = knnClassify(trainSet,trainSet[:,5],testSet[i],k)
        if votedLabel != actual_label:
            errorCount += 1
    dataSize = testSet.shape[0]
    print("============error count is %d================"%(errorCount))
    correct_count = dataSize - errorCount
    print("============correct count is %d================"%(correct_count))
    print("============error accuracy is %.3f============="%(errorCount/dataSize))
    print("=============correct accuracy is %.3f============="%(correct_count/dataSize))
if __name__ == "__main__":
    bg_path = sys.argv[1]
    fg_path = sys.argv[2]
    trainSet,testSet = createTrainingSetAndTestSet(bg_path,fg_path)
    knnTrain(trainSet,testSet,17)




