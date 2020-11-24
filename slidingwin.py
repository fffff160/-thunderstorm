#! /usr/bin/env python
# -*-coding:utf-8-*-
from Parameter import *
import os
from D131 import D131
import datetime
import numpy as np
from multiprocessing import Pool
import h5py


radar_path = '/data/swan/'
save_path = "/data/512*512*5-15/"
label_path = "/data/label_deal_2017"


top = int((Parameter.radar_start_lat - Parameter.north_lat) / 0.01)
bottom = int((Parameter.radar_start_lat - Parameter.south_lat) / 0.01)
left = int((Parameter.west_lon - Parameter.radar_start_lon) / 0.01)
right = int((Parameter.east_lon - Parameter.radar_start_lon) / 0.01)
box_shape = 512


for label_name in os.listdir(label_path):
    radar_data_10 = []

#######找出满足条件的10个雷达值,并获取值
    for i in range(1, 15):
        dt = datetime.datetime.strptime(label_name[:12], '%Y%m%d%H%M')
        dt = dt - datetime.timedelta(minutes=6 * i)
        dt = dt - datetime.timedelta(hours=8)   #label存的是北京时间
        radar_time = dt.strftime('%Y%m%d%H%M%S')
        radar_file = "Z_OTHE_RADAMOSAIC_" + radar_time + ".bin.bz2"
        radar_file_name = radar_path + radar_time[0:4] + "/" + radar_time[0:6] + "/TDMOSAIC/" + radar_file
        print("radar_file_name: ", radar_file_name)
        if os.path.exists(radar_file_name):
            radar_path_dir = os.path.dirname(radar_file_name)
            radar_file = os.path.basename(radar_file_name)
            date_reader = D131(radar_path_dir, radar_file)
            _, _, data = date_reader.decode()
            radar_tmp = data[:, top:bottom + 1, left:right + 1]
            #print(np.shape(radar_tmp))
            radar_data_10.append(radar_tmp)
            #print(len(radar_data_10))
################################################################################
    if len(radar_data_10) < 14:
        print('**********less than 14*********')
        print(label_name)
        
    
    label_name_all = os.path.join(label_path, label_name)
    #print("label_name_all: ",label_name_all)
    lager_label = np.load(label_name_all)
    print(lager_label.shape)
    #print(np.shape(lager_label))
    print(np.sum(lager_label))
    if (np.sum(lager_label)<10):
        continue
    for i in range(0,lager_label.shape[0],100):
        if (i + box_shape > lager_label.shape[0]):  # 行过了
            break
        for j in range(0,lager_label.shape[1],100):
            if(j+box_shape>lager_label.shape[1]):  #列过了
                break
            else:
                if(np.sum(lager_label[i:i+box_shape,j:j+box_shape])>10):
                    print("%%%%%%%%%%%%%%%%%%%%%%")
                    for M in range(1,len(radar_data_10)-3):
                        radar = np.concatenate(radar_data_10[M-1:M+4]).reshape(5, 21, 3701, 2801)
                        data_save = radar[:, :, i:i + box_shape, j:j + box_shape]     #取第几层还是要再看??
                        # 时间信息
                        t = np.arange(M*6, M*6+5*6, 6).reshape(5, 1, 1, 1)
                        time_arr = t * np.ones((5, 1, 512, 512))
                        # time_arr = np.ones((1, box_shape, box_shape)) * (M*6)
                        # 5*21*512*512 + 5*1*512*512 = 5*22*512*512
                        data_save = np.append(data_save, time_arr, axis=1)
                        print('data shape:', data_save.shape)
                        
                        label_save = lager_label[i:i + box_shape, j:j + box_shape]
                        # label_name = os.path.basename(label_path)
                        # print(save_path + label_name[:-4] + str(M*6)+str(i)+"_"+str(j)+ '.h5')
                        with h5py.File(save_path + label_name[:-4] +'_' + str(M*6) + '_' +str(i)+"_"+str(j)+ '.h5', 'w') as f:
                            f.create_dataset('data', data = data_save, compression = 'gzip', compression_opts = 4)
                            f.create_dataset('label', data = label_save, compression = 'gzip', compression_opts = 4)
                        print("save data")
                        print(save_path + label_name[:-4] +'_' + str(M*6) + '_' +str(i)+"_"+str(j)+ '.h5')

