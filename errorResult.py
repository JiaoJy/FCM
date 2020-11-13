import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from numpy import random


def jdwucha(A1,data2):
    dist_temp = np.abs(A1 - data2) / (data2.shape[0])
    dist = dist_temp.sum(axis=0)
    return dist

def xdwucha(data1,data2):                 #data1是预测的矩阵；data2是要做比较的真实矩阵
    dist_temp = (np.abs((data1 - data2)/data2))/(data2.shape[0])
    dist = dist_temp.sum(axis=0)
    return dist

def MSE(A1,data2):
    dist_temp = (np.power((A1 - data2),2))/data2.shape[0]
    dist = dist_temp.sum(axis=0)
    return dist

def RMSE(M):
    dist = np.power(M,0.5)
    return dist

def fanguiy(A1):#反归一化
    A2 = A1.copy()
    for j in range(0,A1.shape[1]):
        madata = max(data[:, j])
        midata = min(data[:, j])
        A2[:,j] = A1[:,j] * (madata - midata) + midata

    return A2

def drawPre(title,preData,realData,dataNum=10):
    plt.title(title)
    plt.plot(range(dataNum), preData, color='green', label='predict');
    plt.plot(range(dataNum), realData, color='red', label='real');
    plt.legend();
    plt.xlabel('time');
    plt.ylabel('value');
    plt.show();

def drawAll(data_pre,data_test):
    drawPre("CO",data_pre[:,0],data_test[:,0],dataNum = data_test.shape[0])
    drawPre("NO2",data_pre[:,1],data_test[:,1],dataNum = data_test.shape[0])
    drawPre("SO2",data_pre[:,2],data_test[:,2],dataNum = data_test.shape[0])
    drawPre("O3",data_pre[:,3],data_test[:,3],dataNum = data_test.shape[0])
    drawPre("PM25",data_pre[:,4],data_test[:,4],dataNum = data_test.shape[0])
    drawPre("PM10",data_pre[:,5],data_test[:,5],dataNum = data_test.shape[0])
    
def result(data_pre,data_test,path):
    path = 'error_Result/result_prediction_'+path+'.csv'
    Mwucha = MSE(data_pre,data_test)
    Rwucha = RMSE(Mwucha)
    Bm = np.linalg.norm(data_pre-data_test)/data_pre.shape[0]
    jd = jdwucha(data_pre,data_test)
    xd = xdwucha(data_pre,data_test)
    print('绝对误差：{}'.format(jd))
    print('相对误差：{}'.format(xd))
    print('欧式距离：{}'.format(Bm))
    print('MSE：{}'.format(Mwucha))
    print('RMSE:{}'.format(Rwucha))
    temp = [0]*6
    temp[0] = ["CO","NO2","SO2","O3","PM25","PM10"]
    temp[1] = jd
    temp[2] = xd
    temp[3] = Mwucha
    temp[4] = Rwucha
    temp[5] = [Bm,0,0,0,0,0]
    temp = np.array(temp)
    temp = pd.DataFrame(temp)
    temp.to_csv(path,header=None)
    
def result_alt(data_pre,data_test,path):
    path = 'error_Result/result_prediction_'+path+'.csv'
    Mwucha = MSE(data_pre,data_test)
    Rwucha = RMSE(Mwucha)
    Bm = np.linalg.norm(data_pre-data_test)/data_pre.shape[0]
    jd = jdwucha(data_pre,data_test)
    xd = xdwucha(data_pre,data_test)
    print('绝对误差：{}'.format(jd))
    print('相对误差：{}'.format(xd))
    print('欧式距离：{}'.format(Bm))
    print('MSE：{}'.format(Mwucha))
    print('RMSE:{}'.format(Rwucha))
    temp = [0]*6
    temp[0] = ["C1","C2","C3","C4","C5"]
    temp[1] = jd
    temp[2] = xd
    temp[3] = Mwucha
    temp[4] = Rwucha
    temp[5] = [Bm,0,0,0,0]
    temp = np.array(temp)
    temp = pd.DataFrame(temp)
    temp.to_csv(path,header=None)

def drawAll_alt(data_pre,data_test):
    drawPre("C1",data_pre[:,0],data_test[:,0],dataNum = data_test.shape[0])
    drawPre("C2",data_pre[:,1],data_test[:,1],dataNum = data_test.shape[0])
    drawPre("C3",data_pre[:,2],data_test[:,2],dataNum = data_test.shape[0])
    drawPre("C4",data_pre[:,3],data_test[:,3],dataNum = data_test.shape[0])
    drawPre("C5",data_pre[:,4],data_test[:,4],dataNum = data_test.shape[0])