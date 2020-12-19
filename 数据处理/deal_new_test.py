#! /usr/bin/python
# -*-coding:utf-8-*- 
import os
import numpy as np
from Parameter import *
import os
from D131 import D131
import datetime
from datetime import datetime, timedelta
import h5py


top = int((Parameter.radar_start_lat - Parameter.north_lat) / 0.01)
bottom = int((Parameter.radar_start_lat - Parameter.south_lat) / 0.01)
left = int((Parameter.west_lon - Parameter.radar_start_lon) / 0.01)
right = int((Parameter.east_lon - Parameter.radar_start_lon) / 0.01)

label_path = '/data/label_deal_2017/'
radar_path = '/data/swan/'
save_path = '/data/wind_test_data_new/'
box_shape = 512

conitue_data = 10  #连续几张



def deal_radar_name(file):
    radar_flag = 0
    radars_name_all = []
    for i in range(5,0,-1):
        radar_t_pre = datetime.strptime(file[:-6], '%Y%m%d%H%M') + timedelta(minutes=-6 * i)-timedelta(hours=8)
        radar_time = datetime.strftime(radar_t_pre, '%Y%m%d%H%M%S')
        radar_file = "Z_OTHE_RADAMOSAIC_" + radar_time + ".bin.bz2"
	#print(radar_file)
        radar_file_name = radar_path + radar_time[0:4] + "/" + radar_time[0:6] + "/TDMOSAIC/" + radar_file
	#print(radar_file_name)
        if os.path.exists(radar_file_name):
	   # print(radar_flag)
            radar_flag += 1
	    #print(radar_flag)
            radars_name_all.append(radar_file_name)
    return radar_flag,radars_name_all


def deal_radar_data(radar_files):
    radar_data_all = []
    for radar_file in radar_files:
        radar_path_dir = os.path.dirname(radar_file)
        radar_file = os.path.basename(radar_file)
        date_reader = D131(radar_path_dir, radar_file)
        _, _, data = date_reader.decode()
        radar_tmp = data[:, top:bottom + 1, left:right + 1]
        radar_data_all.append(radar_tmp)
    return  np.array(radar_data_all)



def deal_label_name(file):
    label_flag = 0
    labels_name_all = []
    for i in range(conitue_data):
        label_t_next = datetime.strptime(file[:-6], '%Y%m%d%H%M') + timedelta(minutes= 6 * i)
        label_time = datetime.strftime(label_t_next, '%Y%m%d%H%M%S')
        # print(label_time)
        label_file = label_time + ".npy"
        label_file_name = label_path + label_file
        if os.path.exists(label_file_name):
            label_flag += 1
            labels_name_all.append(label_file_name)
        else:
            labels_name_all.append('zero_pad')

    return label_flag, labels_name_all



def deal_label_data(label_files):
    label_data_all = []
    for label_file in label_files:
        if label_file == 'zero_pad':
            label_data_all.append(np.zeros((3701, 2801), dtype=np.bool))
        else:
            wind = np.load(label_file)
            label_data_all.append(wind)

    return np.array(label_data_all)




if __name__ =="__main__":

    label_files = os.listdir(label_path)
    for file in label_files:
	#print(file,"EEEEEE")
        radar_flag, radars_name_all = deal_radar_name(file)
	print("radar_flag", len(radars_name_all))
        label_flag, labels_name_all = deal_label_name(file)
	print("label_flag", label_flag)
	#break
        if len(radars_name_all) == 5 and label_flag >= 6:    ##大区域连续的几张
	    #break
            print("###########")
            label_data_10 = deal_label_data(labels_name_all)
            radar_data_6 = deal_radar_data(radars_name_all)

            for i in range(0, label_data_10.shape[1], 512):
                if (i + box_shape > label_data_10.shape[1]):  
                    break
                for j in range(0, label_data_10.shape[2], 512):
                    if (j + box_shape > label_data_10.shape[2]):  
                        break
                    else:
                        if np.sum(label_data_10[:, i:i + box_shape, j:j + box_shape]) < 10:
                            continue
                      
                        for M in range(1, len(label_data_10) +1):
                            data_save = radar_data_6[:, :, i:i + box_shape, j:j + box_shape]
                            t = np.arange(M * 6, M * 6 + 5 * 6, 6).reshape(5, 1, 1, 1)
                            time_arr = t * np.ones((5, 1, 512,  512))
                            data_save = np.append(data_save, time_arr, axis=1)
                            print('data shape:', data_save.shape)

                            label_save = label_data_10[M-1, i:i + box_shape, j:j + box_shape]    
                            # label_name = os.path.basename(label_path)
                            # print(save_path + label_name[:-4] + str(M*6)+str(i)+"_"+str(j)+ '.h5')
                            save_file = datetime.strptime(file[:-6], '%Y%m%d%H%M') + timedelta(minutes= 6 * (M-1))
                            save_file_name = datetime.strftime(save_file, '%Y%m%d%H%M%S')
                            
                            with h5py.File(
                                    save_path + save_file_name + '_' + str(M * 6) + '_' + str(i) + "_" + str(
                                            j) + '.h5', 'w') as f:
                                f.create_dataset('data', data=data_save, compression='gzip', compression_opts=4)
                                f.create_dataset('label', data=label_save, compression='gzip', compression_opts=4)
                            print("save data")
                            print(save_path + file[:-4] + '_' + str(M * 6) + '_' + str(i) + "_" + str(
                                j) + '.h5')













