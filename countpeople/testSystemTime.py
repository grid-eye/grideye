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
path = "double_sensor"
if len(sys.argv) > 3:
    path = sys.argv[3]
if not os.path.exists(path):
    os.mkdir(path)
if len(sys.argv) > 4:
    show_arg = sys.argv[4]
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
mythread2 = myThread(host2,port2,con,event)
mythread1.start()
mythread2.start()
event.set()
print(" is start receive sensor data ? ",end = ":")
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
i = 0 
container = []
try:
    while True:
        if mythread1.getQuitFlag() or mythread2.getQuitFlag():
            break
        i += 1
        print(" the %dth frame "%(i))
        print("============wait=============")
        s1 = mythread1.getNextFrame()
        s2 = mythread2.getNextFrame()
        print(s1,end="======>")
        print(s2)
        time.sleep(1)