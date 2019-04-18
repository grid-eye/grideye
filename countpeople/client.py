import socket
import sys
import numpy as np
import pickle
import time
import threading
import os
import cv2 as cv
from countpeople import CountPeople
socket1 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
socket2  = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host1 = "192.168.1.100"
host2 = "192.168.1.211"
port1 = 9999
show_frame = False
port2 = port1
all_frame_sensor_1 = []
all_frame_sensor_2 = []
if len(sys.argv) > 1:
    port1 = int(sys.argv[1])
    port2 = port1
    if len(sys.argv) > 2:
        port2 = int(sys.argv[2])
path = "double_sensor"
if len(sys.argv) > 3:
    path = sys.argv[3]
if not os.path.exists(path):
    os.mkdir(path)
if len(sys.argv) > 4:
    show_arg = sys.argv[4]
    if show_arg == "show_frame":
        show_frame = True 
class myThread (threading.Thread) :
    def __init__(self,threadID,name,lock,container,socket,condition):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.lock = threading.Lock()
        self.container = container
        self.socket = socket
        self.condition = condition
        self.quit = False
    def setQuitFlag(self,flag):
        self.quit = True
    def getQuitFlag(self):
        return self.quit
    def run(self):
        print("start thread "+self.name)
        try:
            while True:
                if self.condition.acquire():
                    if len(self.container) == 0:
                        self.condition.wait()
                    recv = self.socket.recv(1024)
                    data = pickle.loads(recv)
                    recv = np.array(recv)
                    self.socket.send("ok".encode("utf-8"))
                    self.container.append(data)
                    self.condition.notify()
                    self.condition.wait(3)
                    self.condition.release()
        except KeyboardInterrupt:
            print("keyboardinterrupt ..........")
            self.setQuitFlag = True
    def getContainer(self):
        return self.container
data_container = [ ]
socket1.connect((host1,port1))
socket2.connect((host2,port2))
lock = threading.Lock()#互斥锁
con = threading.Condition()#为了轮流读取两个服务器的数据,不需要互斥锁了
res = con.acquire()#提前让主线程获得锁
if not res :
    raise RuntimeError()
mythread = myThread("001","wangThread",lock,data_container,socket2,con)
mythread.start()
socket1.settimeout(3)
socket2.settimeout(3)
def showData(data):
    for item in data:
        print(np.array(item))
    print("================")

i = 0 
thresh = 40
def saveImageData(sensor1,sensor2,path):
    np.save(path+"/sensor1.npy",np.array(sensor1))
    np.save(path+"/sensor2.npy",np.array(sensor2))
def mergeData(t1,t2):
    temp = np.zeros(t1.shape)
    print(" t1 shape is")
    print(t1.shape)
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            temp[i][j] = max(t1[i][j],t2[i][j])
    return temp
all_merge_frame = []
cp = CountPeople()
try:
    while True:
        if i > 0:
            con.acquire()
        if mythread.getQuitFlag():
            break
        i += 1
        print(" the %dth frame "%(i))
        msg = socket1.recv(1024)
        msg = pickle.loads(msg)
        msg = np.array(msg)
        socket1.send("ok".encode("utf-8"))
        all_frame_sensor_1.append(msg)
        data_container.append(msg)
        con.notify()
        con.wait(3)
        all_frame_sensor_2.append(np.array(data_container[1]))
        showData(data_container)
        s1,s2 = data_container
        data_container.clear()
        con.release()
        current_frame = mergeData(s1,s2)#合并两个传感器的数据,取最大值
        if not cp.isCalcBg(): 
            if i == thresh:
                avgtemp = cp.calAverageTemp(all_merge_frame)
                cp.setCalcBg(True)
                cp.setBgTemperature(avgtemp)
                cp.constructBgModel(avgtemp)
                if show_frame:
                    cv.nameWindow("image",cv.WINDOW_NORMAL)
                cp.calcBg = True
                all_merge_frame=[]
            else:
                all_merge_frame.append(current_frame)
            continue
        if show_frame:
            plot_img = np.zeros(current_frame.shape,np.int8)
            plot_img[ np.where(diff > 1.5) ] = 255
            img_resize  = cv.resize(plot_img,(16,16),interpolation=cv.INTER_CUBIC)
            cv.imshow("image",img_resize)
            cv.waitKey(1)
        res = False
        diff = current_frame - avgtemp
        ret = cp.isCurrentFrameContainHuman(current_frame,avgtemp,diff)
        if not ret[0]:
            cp.updateObjectTrackDictAgeAndInterval()
            cp.tailOperate(current_frame)
            if cp.getExistPeople():
                cp.setExistPeople(False)
            continue
        cp.setExistPeople(True)
        print("extractbody")
        (cnt_count,image ,contours,hierarchy),area =cp.extractBody(cp.average_temp, current_frame)
        if cnt_count ==0:
            cp.updateObjectTrackDictAgeAndInterval()
            cp.tailOperate(current_frame)
            continue
        #下一步是计算轮当前帧的中心位置
        loc = cp.findBodyLocation(diff,contours,[ i for i in range(cp.row)])
        cp.trackPeople(current_frame,loc)#检测人体运动轨迹
        cp.updateObjectTrackDictAge()#增加目标年龄
        cp.tailOperate(current_frame)
        #sleep(0.5)
        if mythread.getQuitFlag():
            break
        if i >= thresh:
            saveImageData(all_frame_sensor_1,all_frame_sensor_2,path)
            thresh += 500 
except KeyboardInterrupt:
    print("==========sensor catch keyboardinterrupt==========")
finally:
    saveImageData(all_frame_sensor_1,all_frame_sensor_2,path)
    mythread.setQuitFlag(True)
    socket1.close()
    socket2.close()
print(" exit sucessfully!")
