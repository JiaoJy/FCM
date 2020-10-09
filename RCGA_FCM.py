import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from numpy import random

def errorLp(W,data,data_front,time):
    A1 = cala(W,data,data_front,time)                 #更新后的矩阵
    dist = np.linalg.norm(A1 - data)/ItemNum1         #预测矩阵与真实矩阵的欧式距离
    return dist

def initialization(pop_size,A,data,data_front):
    B1 = [0]*pop_size                                        #对应误差
    for i in range (pop_size):
        B1[i] = errorLp(A[i], data, data_front ,ItemNum1)                 #计算初始种群的误差

    j = B1[B1.index(min(B1))]
    g = B1.index(min(B1))
    b = B1.index(max(B1))
    bestw= A[g]
    
    return bestw,j,b                                         #返回最好的权重和误差,和最坏的下标

def f(x):
    y = 1/(1+np.exp(-x))
    return y

def cala(W,data,data_front,time):
    A2 = [0] * time
    A2[0] = f(np.dot(data_front,W))
    for i in range(1,time):
        A2[i] = f(np.dot(data[i-1], W))
        #A2[i + 1] = f(temp)
    return A2

def cala1(W,data):
    A2 = np.zeros(data.shape[0])
    for j in range(data.shape[0]):
        temp = np.dot(W[j],data)
        A2[j] = f(temp)
    return A2

def caa(W,data):
    A2 = [0]*a
    for o in range (0,a):
        temp = np.dot(data,W)
        A2 = f(temp)
    return A2
        
def crossover(A):#融合交叉
    #A = list(A)
    nA = [0]*pop_size

    for i in range (0,int(pop_size/2)):
        if random.random() < pop_cr:

            r = random.random()
            R1 = np.array([[r] * ItemNum2 for i in range(ItemNum2)])
            R2 = np.array([[1-r] * ItemNum2 for i in range(ItemNum2)])

            nA[2 * i] = (R1 * np.array(A[2 * i]) + R2 * np.array(A[2 * i+1]))
            nA[2 * i+1] = (R1 * np.array(A[2 * i+1])+ R2 * np.array(A[2 * i]))
        else:
            nA[2 * i] = A[2 * i]
            nA[2 * i+1] = A[2 * i+1]
    return nA

def mutation(A):
    for i in range(0,pop_size):
        A[i] = np.array(A[i])
        if random.random() < pop_mr:
            step = random.uniform(0,0.1)
            step1= np.array([[step] * 6 for i in range(6)])
            if random.random()< 0.1:
                A[i] = A[i] - step1
            else:
                A[i] = A[i] + step1
    return A

def newA(BESTW,A,data,data_front ,k,m):                                          
    #生成新的种群和更新最优权重矩阵
    A = crossover(A)
    A = mutation(A)

    BESTWtemp,j,p = initialization(pop_size, A, data, data_front)
    if j<=k:
        BESTW = BESTWtemp  #留下每一代最好的权重矩阵
        k = j
    A[p] = BESTW
    return A,BESTW,k

arr1 = pd.read_csv('dataProcess.csv',index_col = 0)
data1 = np.array(arr1)
if __name__ == '__main__':
    pop_size = 8  # 初始种群个数
    pop_cr = 0.7  # 交叉概率
    pop_mr = 0.3  # 变异概率
    pop_ma = 1  # 变异系数
    iternum = 100
    w = 504        #滑动窗口大小
    a = 1      #预测未来一小时的
    start = 0
    dataf = pd.read_csv('dataProcess.csv',index_col = 0)
    data2 = dataf.iloc[w + start:w + iternum + start,:]

    Num1=data2.shape[0]                             #测试集行数100
    Num2=data2.shape[1]                             #列数6

    data_pred = [0]*iternum
    
    data = data1[start:start+ w, :]    
    data_front = data1[start-1]
    ItemNum1 = data.shape[0]  # 训练集行数504
    ItemNum2 = data.shape[1]  # 列数6
    
    ed = [0]*800
    #训练
    A = [0] * pop_size  # 初始种群
    for j in range(pop_size):
        #A[j] = -1 + 2 * np.random.random((ItemNum2, ItemNum2))
        A[j] = -1 + 2 * np.random.uniform(0,1,(ItemNum2, ItemNum2))
    
    bestw, k, m = initialization(pop_size, A, data, data_front)
    for op in range(800):
        A, bestw,k = newA(bestw, A, data, data_front, k, m)
        ed[op] = k
        
    #预测
    for i in range(0,iternum):
        #t = 0
        #BESTW = [0]
        '''
        #print(i)
        data = data1[i+start:i +start+ w, :]
        
        data_front = data1[i+start-1]
        #print(data_front)
        ItemNum1 = data.shape[0]  # 训练集行数504
        ItemNum2 = data.shape[1]  # 列数6

        A = [0] * pop_size  # 初始种群
        for j in range(pop_size):
            #A[j] = -1 + 2 * np.random.random((ItemNum2, ItemNum2))
            A[j] = -1 + 2 * np.random.uniform(0,1,(ItemNum2, ItemNum2))
        
        bestw, k, m = initialization(pop_size, A, data, data_front)
        for op in range(500):
            A, bestw = newA(bestw, A, data, data_front, k, m)
        '''
        
        data_test = data1[i + start + w-1]  # 测试输入
        data_pred[i] = cala1(bestw,data_test)
    #print(bestw)                  # 最优权重矩阵展示

    #最优权重矩阵做预测
    data_pre = np.array(data_pred)                    #预测出的数据
    data_test = np.array(data2)
    #A1 = fanguiy(A1)                           #反归一化后的预测数据

    drawAll(data_pre,data_test)
    result(data_pre,data_test,'rcga')