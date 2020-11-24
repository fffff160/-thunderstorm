import numpy as np
import matplotlib.pyplot as plt
from Parameter import *
import os
from D131 import D131
import datetime


top = int((Parameter.radar_start_lat - Parameter.north_lat) / 0.01)
bottom = int((Parameter.radar_start_lat - Parameter.south_lat) / 0.01)
left = int((Parameter.west_lon - Parameter.radar_start_lon) / 0.01)
right = int((Parameter.east_lon - Parameter.radar_start_lon) / 0.01)

print(top, bottom, left, right)
print(bottom-top+1)
print(right-left+1)

fold_path = r'E:\work\new_work\雷暴大风\swan'
save_path = 'E:\\work\\new_work\\data\\radar\\'

files = os.listdir(fold_path)
for file in files:
    try:
        utc_time = file.split('.')[0].split('_')[3]
        print(utc_time)
        dt = datetime.datetime.strptime(utc_time, '%Y%m%d%H%M%S')
        dt = dt - datetime.timedelta(hours=8)
        beijing_time = dt.strftime('%Y%m%d%H%M%S')
        print(beijing_time)

        file_name = save_path + beijing_time + '.npy'
        print(file_name)

        if os.path.exists(file_name):
            continue

        date_reader = D131(fold_path, file)
        _, _, data = date_reader.decode()
        data = data[:, top:bottom+1, left:right+1]
        data[data<0] = 0
        data[data>70] = 70


        np.save(file_name, data)
        del date_reader
        del data
    except:
        pass
    