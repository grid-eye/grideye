import socket
import sys
import numpy as np
import pickle
import time
import threading
import os
import cv2 as cv
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s2 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host1 = "192.168.1.100"
host2 = "192.168.1.211"
port1 = 9999
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
                        self.wait()
                    recv = self.socket.recv(1024)
                    data = pickle.loads(recv)
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
s.connect((host1,port1))
s2.connect((host2,port2))
lock = threading.Lock()#互斥锁
con = threading.Condition()#为了轮流读取两个服务器的数据,不需要互斥锁了
res = con.acquire()#提前让主线程获得锁
if not res :
    raise RuntimeError()
mythread = myThread("001","wangThread",lock,data_container,s2,con)
mythread.start()
s.settimeout(3)
s2.settimeout(3)
def showData(data):
    for item in data:
        print(np.array(item))
    print("================")

i = 0 
thresh = 1000
def saveImageData(sensor1,sensor2,path):
    np.save(path+"/sensor1.npy",np.array(sensor1))
    np.save(path+"/sensor2.npy",np.array(sensor2))
try:
    while True:
        if not res :
            con.acquire()
        if mythread.getQuitFlag():
            break
        i += 1
        print(" the %dth frame "%(i))
        msg = s.recv(1024)
        msg = pickle.loads(msg)
        s.send("ok".encode("utf-8"))
        all_frame_sensor_1.append(np.array(msg))
        data_container.append(msg)
        con.notify()
        con.wait(3)
        all_frame_sensor_2.append(np.array(data_container[1]))
        showData(data_container)
        data_container.clear()
        con.release()
        res = False
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
    s.close()
    s2.close()
print(" exit sucessfully!")
