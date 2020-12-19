import time
import datetime
import os

fold_path = ['F:\\雷暴大风资料\\dealed_flash_data\\2016\\','F:\\雷暴大风资料\\dealed_flash_data\\2017\\','F:\\雷暴大风资料\\dealed_flash_data\\2018\\']
flash_path_all = [r"\\10.28.21.164\lightning\2016",r"\\10.28.21.164\lightning\2017",r"\\10.28.21.164\lightning\2018"]
i = 0
for flash_path in flash_path_all:
    files = os.listdir(flash_path)
    for file in files:
        with open(flash_path +"\\" + file) as f:
            lines = f.readlines()
            for line in lines:
                fields = line.split()
                t = fields[1] + ' ' + fields[2][:8]
                dt = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                print(dt)
                stamp = time.mktime(dt.timetuple())
                dt = datetime.datetime.fromtimestamp(int(stamp+5*60)//600*600)
                print(dt)
                file_name = dt.strftime('%Y%m%d%H%M%S')
                print(file_name)
                with open(fold_path[i] + file_name + '.txt', 'a') as ff:
                    ff.writelines(line)
    i += 1
