# coding=utf-8

import csv
import numpy as np
import pandas as pd
import os
from collections import Counter
import datetime
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 读取文件数据
def read_file(path,name,date_column=9,index=9,header=0):
    file_url=os.path.join(path,name)
    data = pd.read_csv(file_url,parse_dates=[date_column],index_col=index)
    #DATA = DATA[~DATA.index.duplicated()]   # 去除重复索引
    return data

# 重新设置索引
def reindex_datetime(data,freq="H"):
    data = data[~data.index.duplicated()]  # 去除重复索引
    rs = pd.date_range(data.index[0],data.index[-1],freq=freq) #生成时间索引(参数start、end)
    return data.reindex(rs)

# 空缺位置补均值
def imputer(data):
    data.columns = list(range(data.shape[1]))  # 列索引
    x = data.rolling(7,min_periods=1) # 时间滑动窗口
    xc = x.count()
    xm = x.mean()
    id = data.index
    # print(DATA.isnull().sum())

    for i in id:
        # print(xc.loc[str(i), 0])
        if np.isnan(data.loc[str(i),0]):
            data.loc[str(i)] = xm.loc[str(i)]

    return data.dropna()

# 数据归一化
def normalization(data):
    return (data-data.min())/(data.max()-data.min())

# 数据标准化
def standardization(data):
    return (data-data.mean())/(data.std())

# 数据异常值处理
def outlier(data):
    x = data.rolling(window=15, min_periods=1, center=True)
    box = pd.concat([x.quantile(0.25), x.quantile(0.75)], axis=1)
    a = data.columns.size
    box.columns = list(range(a * 2))
    for i in range(a):
        data.loc[data[i] > box[a + i] + 1.5 * (box[a + i] - box[i]), i] = box.loc[data[i] > box[a + i] + 1.5 * (box[a + i] - box[i]), a + i]
        data.loc[data[i] < box[i] - 1.5 * (box[a + i] - box[i]), i] = box.loc[data[i] < box[i] - 1.5 * (box[a + i] - box[i]), i]
    return data

def dataPreprocess():
    str_path = "E:\Python_Codes\Data\数据20171205\数据\沧州渤海临港产业园"
    file2 = 'XH8082015110300850.csv'

    # 读取数据
    data = read_file(str_path, file2).iloc[:, [1,2,3,4,5,6,7,8]]
    # DATA.columns = ['CO','NO2','SO3','O3','PM25','PM10','TEMPERATURE','HUMIDITY']

    # 数据加工
    data = reindex_datetime(data)
    data = imputer(data)
    data = outlier(data)
    data = standardization(data)
    data.columns = ['CO', 'NO2', 'SO3', 'O3', 'PM25', 'PM10', 'TEMPERATURE', 'HUMIDITY']

    # 输出图表
    # plt.rcParams['font.sans-serif'] = ['KaiTi']
    # plt.rcParams['font.serif'] = ['KaiTi']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # plt.xlabel('时间（小时）')
    # plt.ylabel('标准化后的数据')
    #
    # plt.plot(DATA.iloc[:, 0], label='CO')
    # plt.plot(DATA.iloc[:, 1], label='NO2')
    # plt.plot(DATA.iloc[:, 2], label='SO3')
    #
    # plt.legend()
    # plt.show()

    return data.iloc[:, 0]
    # return DATA






