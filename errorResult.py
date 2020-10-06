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