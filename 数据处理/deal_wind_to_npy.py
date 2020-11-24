'''
    观测站数据转ndarry
'''
from MDFSPlotData import MDFSPlotData
import numpy as np
import matplotlib.pyplot as plt
import os
from Parameter import *

datapath = r"\\10.28.21.164\aws_data\2016\SURFACE\PLOT_10MIN"
savepath = 'D:\\雷暴大风处理后\\wind\\'
# file2 = "20160309190000.000"

rows = round(round(Parameter.north_lat - Parameter.south_lat, 2) / 0.01) + 1
columns = round(round(Parameter.east_lon - Parameter.west_lon, 2) / 0.01) + 1

print(rows)
print(columns)

files = os.listdir(datapath)
print(len(files))
#temp1 = []
for file in files:

    wind_data = np.zeros((rows, columns))
    
    with open(datapath + "\\" +file, 'rb') as f:
        data = f.read()
        MDFSdata = MDFSPlotData()
        MDFSdata.readFromloadByteArray(data)
        MDFSdata.stationDataToPlotDataDict()
        
        for record in MDFSdata.stationDataList:
            if '211' in record[1]:
                val = record[1]['211']
                lon = record[1]['1']
                lat = record[1]['2']

                if lat < Parameter.south_lat or lat > Parameter.north_lat or lon < Parameter.west_lon or lon > Parameter.east_lon:
                    continue
                    
                x = round(round(Parameter.north_lat - lat, 2) / 0.01)
                y = round(round(lon - Parameter.west_lon, 2) / 0.01)
                
                #print(x, y)
                
                if val >= Parameter.hwind_threshold:
                    print(x, y)
                    top = x-5 if x-5 >= 0 else 0
                    bottom = x+5 if x+5 < rows else rows-1
                    left = y-5 if y-5 >= 0 else 0
                    right = y+5 if y+5 < columns else columns-1
                    print(top, bottom, left, right)
                    wind_data[top:bottom+1, left:right+1] = 1
                
    '''
    print(np.sum(wind_data))
    plt.imshow(wind_data)
    plt.show()
    '''
    file_name = savepath + file[:-4] + '.npy'
    print(file_name)
    np.save(file_name, wind_data.astype(np.bool))
    
    