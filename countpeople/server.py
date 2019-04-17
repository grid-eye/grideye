import time
import numpy as np
import pickle
import busio
import board
import adafruit_amg88xx
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
    i2c = busio.I2C(board.SCL, board.SDA)
    amg = adafruit_amg88xx.AMG88XX(i2c,0x69)
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
        try:
            while True:
                temp = [] 
                for row in amg.pixels:
                    # Pad to 1 decimal place
                    temp += row
                i += 1
                print(" the %dth frame "%(i))
                print(np.array(temp))
                serial_temp = pickle.dumps(temp)
                clientSocket.send(serial_temp)
                rec = clientSocket.recv(30)
                msg = rec.decode("utf-8")
                temp = []
        except (KeyboardInterrupt,ConnectionResetError):
            print("error ............ ")
            break
finally:
    clientSocket.close()
