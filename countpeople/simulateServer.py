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
    serverSocket.close()
    clientSocket.close()
