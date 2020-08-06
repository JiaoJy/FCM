# coding=utf-8

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

# 读取文件数据
def read_file(path,name,date_column=9,index=9,header=0):
    file_url=os.path.join(path,name)
    data = pd.read_csv(file_url,parse_dates=[date_column],index_col=index)
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
    xm = x.mean()
    
    for i in range(data.shape[0]):
        for j in data.columns:
            if np.isnan(data.iloc[i,j]):
                data.iloc[i,j] = xm.iloc[i-2,j]
    return data.dropna()

# 数据最大最小归一化
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

def dataPreprocess(path,file):
    # 读取数据
    data = read_file(path, file).iloc[:, [1,2,3,4,5,6]]
    # DATA.columns = ['CO','NO2','SO3','O3','PM25','PM10','TEMPERATURE','HUMIDITY']
    # 数据加工
    data = reindex_datetime(data)
    data = imputer(data)
    data = outlier(data)
    data = normalization(data)
    data.columns = ['CO', 'NO2', 'SO3', 'O3', 'PM25', 'PM10']
    #data.columns = ['CO', 'NO2', 'SO3', 'O3', 'PM25']
    return data
    
def dataPlt(data):
    #输出图表
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['font.serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.xlabel('时间（小时）')
    plt.ylabel('标准化后的数据')
    
    plt.plot(data.iloc[:, 0], label='CO')
    plt.plot(data.iloc[:, 1], label='NO2')
    plt.plot(data.iloc[:, 2], label='SO3')
    
    plt.legend()
    plt.show()

    return data
    # return DATA

if __name__ == "__main__":
    data = dataPreprocess("数据20171205\数据\沧州渤海临港产业园","XH8082015110300850.csv")
    data.to_csv('dataProcessM.csv')
    