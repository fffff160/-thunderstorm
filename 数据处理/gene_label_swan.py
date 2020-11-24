#! /usr/bin/env python
# -*-coding:utf-8-*-
import os
import numpy as np
from Parameter import *
import os
from D131 import D131
import datetime


top = int((Parameter.radar_start_lat - Parameter.north_lat) / 0.01)
bottom = int((Parameter.radar_start_lat - Parameter.south_lat) / 0.01)
left = int((Parameter.west_lon - Parameter.radar_start_lon) / 0.01)
right = int((Parameter.east_lon - Parameter.radar_start_lon) / 0.01)

flash_path = '/data/flash_deal/2016/'
wind_path = '/data/wind_deal/wind/'
radar_path = '/data/swan/'

save_path = '/data/label_deal/'

wind_files = os.listdir(wind_path)
for file in wind_files:
    wind_file_name = wind_path + file
    flash_file_name = flash_path + file

    min = float(file[10:12])
    file = file[:10] + str(int(round(min / 6) * 6)).zfill(2) + '00.npy'
    time = file[:-6]
    dt = datetime.datetime.strptime(time, '%Y%m%d%H%M')
    dt = dt - datetime.timedelta(hours=8)
    radar_time = dt.strftime('%Y%m%d%H%M')
    radar_file = "Z_OTHE_RADAMOSAIC_" + radar_time + "00.bin.bz2"
    radar_file_name = radar_path +radar_time[0:4]+"/"+radar_time[0:6]+"/TDMOSAIC/"+radar_file
    print(radar_file_name)
    label_name = save_path + file
    if os.path.exists(label_name):
        print("exist file")
        continue
    try:
        if os.path.exists(flash_file_name) and os.path.exists(radar_file_name):
            # read wind
            wind = np.load(wind_file_name).astype(np.bool)
    
            # read flash
            flash = np.load(flash_file_name).astype(np.bool)
    
            # read radar
            radar_file = os.path.basename(radar_file_name)
            radar_path_tmp = os.path.dirname(radar_file_name)
            date_reader = D131(radar_path_tmp,radar_file)
            _, _, data = date_reader.decode()
            radar = data[:, top:bottom + 1, left:right + 1]
            radar[radar < 0] = 0
            radar[radar > 70] = 70
    
            radar = np.max(radar, axis=0)
            radar[radar < 35] = 0
            radar[radar >= 35] = 1
    
            radar = radar.astype(np.bool)
    
            label = wind * (flash + radar)
    
            print(np.sum(label))
            if(np.sum(label)== 0):
                continue
            print(label.dtype)
    
            label_name = save_path + file
            print(label_name)
            np.save(label_name, label)
    except:
        print("error file")
        continue

