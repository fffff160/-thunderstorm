from MDFSPlotData import MDFSPlotData
import numpy as np
import matplotlib.pyplot as plt

datapath = r"\\10.28.21.164\aws_data\2017\SURFACE\PLOT_10MIN\\"
file2 = "20160309190000.000"

with open(datapath + file2, 'rb') as f:
    data = f.read()
    MDFSdata = MDFSPlotData()
    MDFSdata.readFromloadByteArray(data)
    MDFSdata.stationDataToPlotDataDict()
    temp1 = []
    dict1 = {}
    for record in MDFSdata.stationDataList:
        if '211' in record[1]:
            temp1.append([record[1]['1'], record[1]['2'], record[1]['211']])

    data = np.array(temp1)
    x = data[:, 0]
    y = data[:, 1]
    c = data[:, 2]
    c[c < 17.2] = 0
    c[c >= 17.2] = 1
    plt.scatter(x, y, s=2, c=c)
    plt.show()