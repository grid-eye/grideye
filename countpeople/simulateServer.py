import time
import numpy as np
import pickle
import socket
import threading
import sys
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
serverSocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host = "localhost"
port = 9998
if len(sys.argv) > 1:
    port = int(sys.argv[1])
addr = (host,port)
print(addr)
try:
    serverSocket.bind(addr)
    print("bind the addr %s , %d "%(host,port))
    print("current ip is %s"%(get_host_ip()))
    serverSocket.listen(2)
    print("listenning...")
    while True:
        clientSocket,addr = serverSocket.accept()
        print(addr)
        i = 0 
        q  = np.zeros((8,8))
        temp = q
        try:
            while True:
                i += 1
                time.sleep(0.09)
                timestamp = time.time()
                data = (temp,round(timestamp,1))
                serial_temp = pickle.dumps(data)
                print("simulate the %dth frame "%(i))
                print(np.array(temp))
                clientSocket.send(serial_temp)
                rec = clientSocket.recv(30)
                msg = rec.decode("utf-8")
                print(msg)
        except (KeyboardInterrupt,ConnectionResetError,BrokenPipeError):
            print("error ............ ")
            print("bind the addr %s , %d "%(host,port))
            serverSocket.listen(2)
            print("listenning...")
finally:
    serverSocket.close()
