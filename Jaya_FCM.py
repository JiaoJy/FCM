# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import copy

def f(lmd,c):
    y = 1 / (1 + np.exp(-1*c))
    return y

def h(err):
    y = 1 / (2 * err + 1)
    return y

def errorLp(p,data_pre,data_real):
    dist = np.linalg.norm(data_pre-data_real,ord=p)/(data_pre.shape[0]*data_pre.shape[1])
    return dist
  

#返回初始化权重参数   NxN 
def initialize_parameters_he(npop,N):
    parameters = np.zeros((npop,N*N))
    for i in range(npop):
        parameter = np.random.randn(N*(N-1)) * np.sqrt(2.0 / (N*(N-1)))
        lmd = np.random.uniform(1,3,N)
        parameters[i] = parameter.tolist()+lmd.tolist()
    return parameters
    

def rshape(X,N):
    temp = copy.copy(X)
    param = temp.reshape(N,N) 
    e = np.zeros((N,N))  
    lmd = param[N-1,:]
    for i in range(N):
        num = 0
        for j in range(N):
            if i is not j:
                e[i,j] = param[i,num] 
                num = num + 1
    return e,lmd
    
def jayaTrain(c_data,c_real,time,N,npop=4):
    X = initialize_parameters_he(npop,N)
    # for n in range(npop):
    #     for i in range(N*N):
    #         X[n,i] = min(X[:,i]) + np.random.random()*(max(X[:,i])-min(X[:,i]))
    fitness = np.zeros(npop)
    error = np.zeros(npop)
    worst= 0
    best = 0
    for num in range(500):
        for n in range(npop):
            e,lmd = rshape(X[n,:],N)
            c_pre = fcm(e,lmd,c_data,time)
            fitness[n] = h(errorLp(2,c_pre,c_real)) 
            error[n] = errorLp(2,c_pre,c_real)
        worst = fitness.tolist().index(min(fitness))
        best = fitness.tolist().index(max(fitness))
        print(worst,best,fitness)
    
        Xx = np.zeros((npop,N*N))
        for n in range(npop):
            Xx[n,:] = X[n,:] + (X[best,:]-np.abs(X[n,:]))*np.random.random()-(X[worst,:]-np.abs(X[n,:]))*np.random.random()
            for i in range(N*N):
                if Xx[n,i] < min(Xx[:,i]):
                    Xx[n,i] = min(Xx[:,i])
                if Xx[n,i] > max(Xx[:,i]):
                    Xx[n,i] = max(Xx[:,i])
            e,lmd = rshape(Xx[n,:],N)
            c_pre = fcm(e,lmd,c_data,time)
            error2 = errorLp(2,c_pre,c_real)
            if error2 <= error[n]:
                X[n,:] = Xx[n,:]
    e,lmd = rshape(X[best,:],N)
    return e,lmd
       
def fcm(e,lmd,data,time):
    c_len = len(data)
    data_pre = np.zeros((time+1,c_len))
    data_pre[0] = data
    for t in range(0,time):
        for i in range(c_len):
            for j in range(c_len):
                if i is not j :
                    data_pre[t+1,i] += e[j,i]*data_pre[t,j]
            data_pre[t+1,i] += data_pre[t,i]
            data_pre[t+1,i] = f(lmd[i],data_pre[t+1,i])
    return data_pre[:time]

#%%
def drawPre(title,preData,realData,dataNum=120):
    plt.title(title)
    plt.plot(range(dataNum), preData, color='green', label='predict');
    plt.plot(range(dataNum), realData, color='red', label='real');
    plt.legend();
    plt.xlabel('time');
    plt.ylabel('value');
    plt.show();        
#%%
if __name__ == "__main__":
    data = pd.read_csv('dataProcess.csv',index_col = 0)
    data = np.array(data)
    data_train = data[240:480,:]   #50天做训练
    data_test = data[480:600,:]     #20天做测试
    e,lmd = jayaTrain(data_train[0,:],data_train,240,6,npop = 8)
    data_pre = fcm(e,lmd,data_test[0,:],120)
    print(errorLp(2,data_pre,data_test))
    drawPre("CO",data_pre[:,0],data_test[:,0])
    drawPre("NO2",data_pre[:,1],data_test[:,1])
    drawPre("SO2",data_pre[:,2],data_test[:,2])
    drawPre("O3",data_pre[:,3],data_test[:,3])
    drawPre("PM25",data_pre[:,4],data_test[:,4])
    drawPre("PM10",data_pre[:,5],data_test[:,5])
    
    