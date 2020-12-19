#! -*- coding:utf-8 -*-
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

from skimage import measure
import matplotlib.colors as colors
import h5py
import os
import random
import datetime
from code.D131 import D131

def colormap():
    cdict = ['#000000', '#00A1F7', '#00EDED', '#00D900', '#009100', '#FFFF00', '#FF9100', '#FF0000',
             '#D70000', '#C10000', '#FF00F1', '#9700B5', '#AD91F7']
    return colors.ListedColormap(cdict)

top = int((56.0 - 54) / 0.01)
bottom = int((56.0 - 17) / 0.01)
left = int((100 - 70) / 0.01)
right = int((128 - 70) / 0.01)

##########################################################
dec_value_path = '/data/wind_train_code_weight/result/best_iteration_92000.txt'
confidence = 0.720

testset_path = '/data/wind_test_data/'
radar_path = '/data/swan/'
test_filelist = natsorted(os.listdir(testset_path))
#########################################################
sample_num = 512*512

print('load dec_value...')
dec_value = np.loadtxt(dec_value_path)[:,0]
pred = np.zeros_like(dec_value)
pred[dec_value<confidence]=1
print('load finish.')

my_cmap = colormap()
clevs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
norm = colors.BoundaryNorm(clevs, len(clevs)-1)

last_label = ''
for i, test_file in enumerate(test_filelist):
    with h5py.File(testset_path + test_file) as fhandle:
        target = fhandle[u'label'][:]

    # time and coor
    time_coor = test_file.split('.')[0]
    label_name, delta, x_index, y_index = time_coor.split('_')
    x_index, y_index = int(x_index), int(y_index)
    print(label_name)
    # read radar
    if label_name != last_label:
        print('read radar')
        dt = datetime.datetime.strptime(label_name[:-2], '%Y%m%d%H%M')
        dt = dt - datetime.timedelta(hours=8)   #label存的是北京时间
        radar_time = dt.strftime('%Y%m%d%H%M%S')
        radar_file = "Z_OTHE_RADAMOSAIC_" + radar_time + ".bin.bz2"
        radar_file_name = radar_path + radar_time[0:4] + "/" + radar_time[0:6] + "/TDMOSAIC/" + radar_file
        print("radar_file_name: ", radar_file_name)

        radar_path_dir = os.path.dirname(radar_file_name)
        radar_file = os.path.basename(radar_file_name)
        date_reader = D131(radar_path_dir, radar_file)
        _, _, data = date_reader.decode()
        radar_val = data[:, top:bottom + 1, left:right + 1]
    # background
    background = radar_val[:, x_index:x_index+512, y_index:y_index+512]
    background = np.max(background, axis=0)
    #background = background[128:512-128, 128:512-128]
    plt.imshow(background, cmap=my_cmap, norm=norm)
    last_label = label_name
    
    # pred
    pred_cur = pred[i*sample_num:i*sample_num+sample_num].reshape(512, 512)
    contours = measure.find_contours(pred_cur, 0.5)
    for contour in contours:
        plt.plot(contour[:,1], contour[:,0], 'w', linewidth=1)
        
    # ground truth
    contours = measure.find_contours(target, 0.5)
    for contour in contours:
        plt.plot(contour[:,1], contour[:,0], 'k', linewidth=1)
        
    plt.grid()
    plt.savefig('./pic/'+radar_time+'_'+delta+'_'+str(x_index)+'_'+str(y_index)+'.png')
    plt.clf()
    
    
    
    
