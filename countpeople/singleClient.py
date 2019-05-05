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
show_frame = False
all_frame_sensor_1 = []
if len(sys.argv) > 1:
    port1 = sys.argv[1]
    split_res = sys.argv[1].split(":")
    if len(split_res) == 2:
        host1 = split_res[0]
        port1 = split_res[1]
    port1 = int(port1)
path = "test"
if len(sys.argv) > 2:
    path = sys.argv[2]
if not os.path.exists(path):
    os.mkdir(path)
if len(sys.argv) > 3:
    show_arg = sys.argv[3]
    if show_arg == "show_frame":
        show_frame = True 
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
                self.socket.send("ok".encode("utf-8"))
                '''
                self.con.acquire()
                self.container.append(recv)
                self.counter += 1
                self.con.notify()
                self.con.release()
                '''
                #self.condition.notify()
                #self.condition.wait(3)
                #self.condition.release()
        except KeyboardInterrupt:
            print("keyboardinterrupt ..........")
            self.setQuitFlag = True
    def getNextFrame(self):
        return self.queue.get() 
        '''
        if self.con.acquire():
            while self.last_cnt >=  self.counter:
                self.con.wait()
            index = self.last_cnt
            self.last_cnt += 1
            self.con.release()
            return self.container[index]
        '''
    def close(self):
        self.socket.close()
lock = threading.Lock()#互斥锁
con = threading.Condition()#为了轮流读取两个服务器的数据,不需要互斥锁了
event = Event()
print(" is start receive sensor data ? ",end = ":")
print(event.is_set())
mythread1 = myThread(host1,port1,con,event)
mythread1.start()
event.set()
print(" is start receive sensor data ? ",end = ":")
def showData(data):
    for item in data:
        print(np.array(item))
    print("================")

i = 0 
thresh = 40
def saveImageData(sensor1,path,avgtemp):
    np.save(path+"/imagedata.npy",np.array(sensor1))
    np.save(path+"/avgtemp.npy",avgtemp)
all_merge_frame = []
cp = CountPeople()
i = 0 
container = []
initial_avg  = None
try:
    while True:
        if mythread1.getQuitFlag() :
            break
        i += 1
        print(" the %dth frame "%(i))
        print("============wait=============")
        s1 = mythread1.getNextFrame()
        time_1 = s1[1]
        s1 = s1[0]
        all_frame_sensor_1.append(s1)
        print("=============show ===========")
        print("=============time is ==============")
        print(time_1)
        time_local_1 = time.localtime(int(time_1))
        dt1 = time.strftime("%Y-%m-%d:%H:%M:%S",time_local_1)
        print(dt1)
        showData([s1])
        current_frame = s1#合并两个传感器的数据,取最大值
        container.append(s1)
        if len(container) == 4:
            last_three_frame = container.pop(0)
        if not cp.isCalcBg(): 
            if i == thresh:
                avgtemp = cp.calAverageTemp(all_merge_frame)
                cp.setCalcBg(True)
                cp.setBgTemperature(avgtemp)
                initial_avg = avgtemp
                cp.constructAverageBgModel(all_merge_frame)
                print(show_frame)
                if show_frame:
                    cv.namedWindow("image",cv.WINDOW_NORMAL)
                cp.calcBg = True
                all_merge_frame=[]
            else:
                all_merge_frame.append(current_frame)
            continue
        diff = current_frame - avgtemp
        if show_frame:
            plot_img = np.zeros(current_frame.shape,np.uint8)
            plot_img[ np.where(diff > 1.5) ] = 255
            img_resize  = cv.resize(plot_img,(16,16),interpolation=cv.INTER_CUBIC)
            cv.imshow("image",img_resize)
            cv.waitKey(1)
        res = False
        ret = cp.isCurrentFrameContainHuman(current_frame,avgtemp,diff)
        if not ret[0]:
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
        loc = cp.findBodyLocation(diff,contours,[ i for i in range(cp.row)])
        cp.trackPeople(current_frame,loc)#检测人体运动轨迹
        cp.updateObjectTrackDictAge()#增加目标年龄
        cp.tailOperate(current_frame,last_three_frame)
        #sleep(0.5)
        if mythread1.getQuitFlag() :
            break
        if i >= thresh:
            saveImageData(all_frame_sensor_1,path,initial_avg)
            thresh += 500 
except KeyboardInterrupt:
    print("==========sensor catch keyboardinterrupt==========")
    saveImageData(all_frame_sensor_1,path,initial_avg)
finally:
    saveImageData(all_frame_sensor_1,path,initial_avg)
    mythread1.setQuitFlag(True)
    mythread1.close()
print(" exit sucessfully!")

