import time
import numpy as np
import pickle
import socket
import threading
import sys
serverSocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host = ""
port = 9998
if len(sys.argv) > 1:
    port = int(sys.argv[1])
addr = (host,port)
try:
    serverSocket.bind(addr)
    print("bind the addr %s , %d "%(host,port))
    serverSocket.listen(2)
    print("listenning...")
    class MyThread(threading.Thread):
        def __init__(self,threadID,name,q):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.q = q 
        def run(self):
            print("start thread : "+self.name)
            self.process_data()
        def process_data(self):
            pass

    while True:
        clientSocket,addr = serverSocket.accept()
        print(addr)
        i = 0 
        q  = np.zeros((8,8))
        temp = q.tolist()
        serial_temp = pickle.dumps(temp)
        try:
            while True:
                i += 1
                time.sleep(0.05)
                print("simulate the %dth frame "%(i))
                print(np.array(temp))
                clientSocket.send(serial_temp)
                rec = clientSocket.recv(30)
                msg = rec.decode("utf-8")
                print(msg)
        except (KeyboardInterrupt,ConnectionResetError):
            print("error ............ ")
            break
finally:
    clientSocket.close()
