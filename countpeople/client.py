import socket
import sys
import numpy as np
import pickle
import time
import threading
from  multiprocessing import Process,Queue ,Event
import os
import cv2 as cv
from countpeople import CountPeople
host1 = "192.168.1.100"
host2 = "192.168.1.211"
port1 = 9999
show_frame = False
port2 = port1
all_frame_sensor_1 = []
sensor_1_original = []
sensor_2_original = []
all_frame_sensor_2 = []
if len(sys.argv) > 1:
    port1 = sys.argv[1]
    split_res = sys.argv[1].split(":")
    if len(split_res) == 2:
        host1 = split_res[0]
        port1 = split_res[1]
        host2 = host1
    port1 = int(port1)
    port2 = port1
    if len(sys.argv) > 2:
        port2 = sys.argv[2]
        split_res = sys.argv[2].split(":")
        if len(split_res) == 2:
            host2 = split_res[0]
            port2 = split_res[1]
        port2 = int(port2)
cp = CountPeople()
path = "double_sensor"
if len(sys.argv) > 3:
    path = sys.argv[3]
if not os.path.exists(path):
    os.mkdir(path)
if len(sys.argv) > 4:
    show_arg = sys.argv[4]
    if show_arg == "show_frame":
        show_frame = True 
merge_shape = (15,8)
cp.setCol(merge_shape[1])
cp.setRow(merge_shape[0])
class myThread (Process) :
    def __init__(self,host,port,condition,event):
        Process.__init__(self)
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.con = condition
        self.quit = False
        self.counter = 0
        self.last_cnt = 0 
        self.queue = Queue(2)
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.socket.connect((self.host,self.port))
        self.event = event
    def setQuitFlag(self,flag):
        self.quit = True
    def getQuitFlag(self):
        return self.quit
    def run(self):
        print("start process")
        print("host2==========")
        print(host2)
        print("port2==========")
        print(port2)
        self.event.wait()
        self.socket.send("start".encode("utf-8"))
        try:
            while True:
                #if len(self.container) == 0:
                #    self.condition.wait()
                recv = self.socket.recv(1024)
                recv = pickle.loads(recv)
                self.counter += 1
                #print("==========the %dth frame========"%(self.counter))
                #print(recv)
                self.queue.put(recv)
                self.socket.send("1".encode("utf-8"))
                #self.condition.notify()
                #self.condition.wait(3)
                #self.condition.release()
        except KeyboardInterrupt:
            print("keyboardinterrupt ..........")
            self.setQuitFlag = True
    def getNextFrame(self):
        return self.queue.get() 
    def close(self):
        self.socket.close()
lock = threading.Lock()#互斥锁
con = threading.Condition()#为了轮流读取两个服务器的数据,不需要互斥锁了
event = Event()
print(" is start receive sensor data ? ",end = ":")
print(event.is_set())
mythread1 = myThread(host1,port1,con,event)
mythread2 = myThread(host2,port2,con,event)
mythread1.start()
mythread2.start()
event.set()
def showData(data):
    for item in data:
        print(np.round(item,2))
        print("================")
i = 0 
thresh = 80#用于计算背景的帧数
diff_time_thresh = 20
def saveImageData(sensor1,sensor2,path,original = None):
    if not original:
        original = ""
    np.save(path+"/"+original+"sensor1.npy",np.array(sensor1))
    np.save(path+"/"+original+"sensor2.npy",np.array(sensor2))
def mergeData(t1,t2,ave=False):
    split = 16-merge_shape[0]
    t1,t2 = t1.copy(),t2.copy()
    row = t1.shape[0]
    sub1 = t1[:split]
    sub2 = t1[-split:]
    temp = np.zeros(sub1.shape)
    for i in range(split):
        for j in range(t1.shape[1]):
            if ave:
                temp[i][j] =round(np.average([ t1[i][j] ,t2[i][j]]),2)
            else:
                temp[i][j] =round(max( t1[i][j] ,t2[i][j]) ,2)
    res = np.append(t2[:-split],temp,axis=0)
    return np.append(res,t1[split:],axis=0)
def isSynchronize(t1,t2,thresh):
    if abs(t1 - t2 ) > thresh:
        return False
    return True
all_merge_frame = []
i = 0 
container = []
time_thresh = 0.02
diff_sum = 0 
toggle = False
align = True#两帧数据时间线是否对齐，即同步
complement = np.load("complement.npy")#传感器之间数据的补偿值
complement_arr = []
s2_arr = []
t = 1
try:
    while True:
        if mythread1.getQuitFlag() or mythread2.getQuitFlag():
            break
        i += 1
        print(" the %dth frame "%(i))
        print("============wait=============")
        s1 = mythread1.getNextFrame()
        s2 = mythread2.getNextFrame()
        showData([s1[0],s2[0]])
        t1 = s1[1]
        t2 = s2[1]
        diff = t1 - t2
        if diff > 0:
            toggle = True#sensor1快
        else:
            toggle = False#sensor2快
        s1= s1[0]
        s2 = s2[0]
        sensor_1_original.append(s1)
        sensor_2_original.append(s2)
        if i < diff_time_thresh:
            diff_sum += abs(diff)
        elif i == diff_time_thresh:
            time_complement = diff_sum / i#计算两个传感器的传送的数据的时间的原始差值,这个差值作为补偿值
            print("======time complement is %.3f "%(time_complement))
            time_thresh += time_complement #判断两个传感器数据是否同步的阈值
            print("======synchronize's thresh is %.3f "%(time_thresh))
        isSync = isSynchronize(t1,t2,time_thresh)
        count = 0
        while_count = 2
        while not isSync and count < while_count:#同步措施
            if not toggle:#sensor2 快
                s1 = mythread1.getNextFrame()
                t1 = s1[1]
                s1 = s1[0]  
                sensor_1_original.append(s1)
            else:
                s2 = mythread2.getNextFrame()#sensor1快
                sensor_2_original.append(s2)
                t2 = s2[1]
                s2 = s2[0]
            diff = t1 - t2 
            if diff >0 :
                toggle = True
            else:
                toggle = False
            isSync = isSynchronize(t1,t2,time_thresh)
            count += 1
        all_frame_sensor_1.append(s1)
        all_frame_sensor_2.append(s2)
        temp = s2
        s2 = s2 + complement#加上补偿值
        if not cp.isCalcBg():
            complement_arr.append(s1-temp)
            s2_arr.append(s2)
        ave = False
        if cp.isCalcBg():
            ret_1 = cp.isCurrentFrameContainHuman(s1,s1_avgtemp,s1-s1_avgtemp)
            ret_2 = cp.isCurrentFrameContainHuman(s2,s2_avgtemp,s2-s2_avgtemp)
            ave = ret_1[0] ^ ret_2[0]
        current_frame = mergeData(s1,s2,ave)#合并两个传感器的数据,取最大值
        print("current_frame is ")
        print(np.round(current_frame,2))
        container.append((s1,s2,current_frame))
        if len(container) == 4:
            last_three_tuple = container.pop(0)
            last_three_frame = last_three_tuple[2]
        if not cp.isCalcBg(): 
            if i == thresh:
                avgtemp = cp.calAverageTemp(np.array(all_merge_frame))
                print(avgtemp)
                s1_avgtemp = cp.calAverageTemp(np.array(all_frame_sensor_1))
                s2_avgtemp = cp.calAverageTemp(np.array(s2_arr))
                cp.setCalcBg(True)
                cp.setBgTemperature(avgtemp)
                cp.constructAverageBgModel(all_merge_frame)
                print("==========time thresh is %.3f============="%(time_thresh))
                complement = np.round(np.average(np.array(complement_arr),axis = 0),2)#更新这个补偿值
                if show_frame:
                    cv.namedWindow("image",cv.WINDOW_NORMAL)
                    cv.namedWindow("sensor1_data",cv.WINDOW_NORMAL)
                    cv.namedWindow("sensor2_data",cv.WINDOW_NORMAL)
                cp.calcBg = True#计算背景完毕
                all_merge_frame=[]
            else:
                all_merge_frame.append(current_frame)
            continue
        diff = current_frame - avgtemp
        diff_bak = diff
        if show_frame:
            plot_img = np.zeros(current_frame.shape,np.uint8)
            plot_img[ np.where(diff > 1.5) ] = 255
            print(plot_img.shape)
            img_resize  = cv.resize(plot_img,(plot_img.shape[1]*3,plot_img.shape[0]*3),interpolation=cv.INTER_CUBIC)
            cv.imshow("image",img_resize)
            cv.waitKey(t)
            plot_img.fill(0)
            diff = s1 - s1_avgtemp
            plot_img = np.zeros(s1.shape,np.uint8)
            print(plot_img.shape)
            shape = (plot_img.shape[0]*4,plot_img.shape[1]*4)
            plot_img[ np.where(diff > 1) ] = 255
            img_resize  = cv.resize(plot_img,shape,interpolation=cv.INTER_CUBIC)
            cv.imshow("sensor1_data",img_resize)
            cv.waitKey(t)
            plot_img.fill(0)
            diff = s2 - s2_avgtemp
            plot_img[ np.where(diff > 1) ] = 255
            img_resize  = cv.resize(plot_img,shape,interpolation=cv.INTER_CUBIC)
            cv.imshow("sensor2_data",img_resize)
            cv.waitKey(t)
        diff = diff_bak
        res = False
        ret = cp.isCurrentFrameContainHuman(current_frame,avgtemp,diff)
        if not ret:
            cp.updateObjectTrackDictAgeAndInterval()
            cp.tailOperate(current_frame,last_three_frame)
            if cp.getExistPeople():
                cp.setExistPeople(False)
            continue
        cp.setExistPeople(True)
        print("extractbody")
        (cnt_count,image ,contours,hierarchy),area =cp.extractBody(cp.average_temp, current_frame)
        if cnt_count ==0:
            cp.updateObjectTrackDictAgeAndInterval()
            cp.tailOperate(current_frame,last_three_frame)
            continue
        #下一步是计算轮当前帧的中心位置
        loc = cp.findBodyLocation(diff,contours)
        cp.trackPeople(current_frame,loc)#检测人体运动轨迹
        cp.updateObjectTrackDictAge()#增加目标年龄
        cp.tailOperate(current_frame,last_three_frame)
        #sleep(0.5)
        if mythread1.getQuitFlag() or mythread2.getQuitFlag():
            break
        if i >= thresh:
            saveImageData(all_frame_sensor_1,all_frame_sensor_2,path)
            saveImageData(sensor_1_original,sensor_2_original,path,original = "original")
            thresh += 500 
except KeyboardInterrupt:
    print("==========sensor catch keyboardinterrupt==========")
finally:
    saveImageData(all_frame_sensor_1,all_frame_sensor_2,path)
    saveImageData(sensor_1_original,sensor_2_original,path,original = "original")
    mythread1.setQuitFlag(True)
    mythread2.setQuitFlag(True)
    mythread1.close()
    mythread2.close()
print(" exit sucessfully!")

