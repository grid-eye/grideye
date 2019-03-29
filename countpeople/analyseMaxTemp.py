import numpy as np
import sys
path = sys.argv[1]
allframe = np.load(path+"/imagedata.npy")
max_temperature = 0
for frame in allframe[:-30]:
    max_temp= frame.max()
    if max_temp > max_temperature:
        max_temperature = max_temp 
print("max temperature is %.2f"%(max_temperature))

    
