#! /usr/bin/env python
# -*-coding:utf-8-*-
'''
Fuction for decode D131
by  guanliang
guanl@cma.gov.cn
2020.06
'''

import numpy as np
import bz2
import os

INT = 'u4'
SHORT = 'u2'
FLOAT = 'f'
LONG = 'u8'

information = np.dtype([('zonname', 'a12'),  # D131
                        ('dataname', 'a38'),  # 数据说明
                        ('flag', 'a8'),  # 文件标志
                        ('version', 'a8'),  # 版本号
                        ('year', SHORT),  # 年
                        ('month', SHORT),  # 月
                        ('day', SHORT),  # 日
                        ('hour', SHORT),  # 时
                        ('minute', SHORT),  # 分
                        ('interval', SHORT)])  # 间隔

gridinfo = np.dtype([('xnum', SHORT),  # 经度格点数
                     ('ynum', SHORT),  # 纬度格点数
                     ('znum', SHORT),  # 层数
                     ('radarcount', INT),  # 拼图雷达数
                     ('startlon', FLOAT),  # 开始经度
                     ('startlat', FLOAT),  # 开始纬度
                     ('centerlon', FLOAT),  # 中心经度
                     ('centerlat', FLOAT),  # 中心纬度
                     ('xreso', FLOAT),  # 经度方向分辨率
                     ('yreso', FLOAT),  # 纬度方向分辨率
                     ('zhighgrids', FLOAT, (40,))])  # 垂直方向的高度

stainfo = np.dtype([('staname', 'a16', (20,)),  # 站点名称
                    ('stalon', FLOAT, (20,)),  # 站点经度
                    ('stalat', FLOAT, (20,)),  # 站点纬度
                    ('staalt', FLOAT, (20,)),  # 站点海拔
                    ('mosaicflag', 'a1', (20,)),  # 该相关站点是否在拼图中
                    ('datatype', SHORT),  # 每一层的向量数
                    ('leveldim', SHORT),
                    ('offset', FLOAT),
                    ('scale', FLOAT)])


class D131():
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def decode(self):
        datapath = self.path + os.sep + self.filename
        f = bz2.BZ2File(datapath, 'rb')
        info = np.frombuffer(f.read(information.itemsize), dtype=information)
        grid = np.frombuffer(f.read(gridinfo.itemsize), dtype=gridinfo)
        sta = np.frombuffer(f.read(stainfo.itemsize), dtype=stainfo)
        res = np.frombuffer(f.read(1024 - (stainfo.itemsize + gridinfo.itemsize + information.itemsize)))
        xnum = grid['xnum']
        ynum = grid['ynum']
        znum = grid['znum']
        lonw = grid['startlon']
        latn = grid['startlat']
        lonreso = grid['xreso']
        latreso = grid['yreso']
        lone = float('%.2f' % (lonw[0] + lonreso[0] * (xnum[0] - 1)))
        lats = float('%.2f' % (latn[0] - latreso[0] * (ynum[0] - 1)))
        lonlist = np.linspace(lonw[0], lone, xnum[0])
        latlist = np.linspace(latn[0], lats, ynum[0])
        databuf = np.frombuffer(f.read(int(znum[0]) * int(ynum[0]) * int(xnum[0])), dtype='u1')
        data3d = databuf.reshape((znum[0], ynum[0], xnum[0])).astype('f4')
        data = (data3d - 66) / 2
        return lonlist, latlist, data


if __name__ == '__main__':
    path = r'I:\STU\python\radar\gl\D131\data'
    filename = 'Z_OTHE_RADAMOSAIC_20200318040000.bin.bz2'
    D131 = D131(path, filename)
    D131.decode()
