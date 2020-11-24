import numpy as np
import matplotlib.pyplot as plt
import os
from Parameter import *

flash_path = "F:\\雷暴大风资料\\dealed_flash_data\\2016\\"
savepath = 'F:\\雷暴大风处理后的资料\\flash\\'

rows = round(round(Parameter.north_lat - Parameter.south_lat, 2) / 0.01) + 1
columns = round(round(Parameter.east_lon - Parameter.west_lon, 2) / 0.01) + 1

print(rows)
print(columns)

files = os.listdir(flash_path)

for file in files:

    flash_data = np.zeros((rows, columns))
    
    with open(flash_path + file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()

            lat = float(line[4].split('=')[1]) # 纬度
            lon = float(line[3].split('=')[1]) # 经度
            
            if lat < Parameter.south_lat or lat > Parameter.north_lat or lon < Parameter.west_lon or lon > Parameter.east_lon:
                    continue
            
            x = round(round(Parameter.north_lat - lat, 2) / 0.01)
            y = round(round(lon - Parameter.west_lon, 2) / 0.01)
            
            flash_data[x, y] = 1
            
    
    file_name = savepath + file[:-4] + '.npy'
    print(file_name)
    np.save(file_name, flash_data.astype(np.bool))
    '''
    data = np.array(data)
    x = data[:,0]
    y = data[:, 1]
    plt.scatter(x, y)
    plt.show()
    '''
        
    