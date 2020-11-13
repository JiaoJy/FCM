# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import copy
import scipy
from errorResult import MSE
from errorResult import RMSE
from errorResult import jdwucha
from errorResult import xdwucha
from errorResult import result
from errorResult import drawAll
from errorResult import fanguiy

def f(lmd,c):
    # return scipy.special.expit(-lmd*c)
    y = 1 / (1 + np.exp(-0.5*c))
    return y

def h(err):
    y = 1 / (2 * err + 1)
    return y

def errorLp(p,data_pre,data_real):
    dist = np.linalg.norm(data_pre-data_real,ord=p)/(data_pre.shape[0]*data_pre.shape[1])
    return dist
  
#%%
#返回初始化权重参数   NxN 
def initialize_parameters_he(npop,N):
    parameters = np.zeros((npop,N*N))
    for i in range(npop):
        parameter = np.random.uniform(-1,1,N*(N-1))
        #np.random.randn(N*(N-1)) * np.sqrt(2.0 / (N*(N-1)))
        lmd = np.random.uniform(0,1,N)
        #lmd = np.array([1]*N)
        parameters[i] = parameter.tolist()+lmd.tolist()
    return parameters
    
def rshape(X,N):
    temp = copy.copy(X)
    e = np.zeros((N,N))
    param = temp[:-N]   
    lmd = temp[-N:]
    num = 0
    for i in range(N):
        for j in range(N):
            if i is not j:
                e[i,j] = param[num]
                num = num + 1
    return e,lmd
    
#jaya学习算法训练FCM参数
def jayaTrain(c_data,c_real,time,N,npop=4):
    X = initialize_parameters_he(npop,N)
    print(rshape(X[1,:],N))
#    for n in range(npop):
#        for i in range(N*N):
#            X[n,i] = min(X[:,i]) + np.random.random()*(max(X[:,i])-min(X[:,i]))
    fitness = np.zeros(npop)
    error = np.zeros(npop)
    worst= 0
    best = 0
    error_time = []
    c_pre = np.zeros((c_real.shape[0],c_real.shape[1]))
    for num in range(1500):
        for n in range(npop):
            e,lmd = rshape(X[n,:],N)
            #print("c_data.shape: %s" % c_data.shape)
            c_pre = fcm(e,lmd,c_data,c_real,time)
            #print("c_pre, c_real: %s ,%s" % c_pre.shape, c_real.shape)
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
            c_pre = fcm(e,lmd,c_data,c_real,time)
            error2 = errorLp(2,c_pre,c_real)
            if error2 <= error[n]:
                X[n,:] = Xx[n,:]              
        error_time.append(error[best])
    e,lmd = rshape(X[best,:],N)
    return e,lmd,error_time

#模糊认知图递推过程      
def fcm(e,lmd,data_front,data,time):
    '''
    c_len = len(data)
    data_pre = np.zeros((time+1,c_len))
    data_pre[0] = data_front
    
    for t in range(0,time):
        for i in range(c_len):
            for j in range(c_len):
                if i is not j :
                    data_pre[t+1,i] += e[j,i]*data[t,j]
            data_pre[t+1,i] += data[t,i]
            data_pre[t+1,i] = f(lmd[i],data[t+1,i])
    
    '''
    c_len = len(data_front)
    data_pre = np.zeros((time,c_len))
    data_pre[0] = np.dot(data_front,e)
    for i in range(c_len):
        data_pre[0,i] = f(lmd[i],data_pre[0,i])
    for t in range(0,time-1):
        data_pre[t+1] = np.dot(data[t],e)
        for i in range(c_len):
            data_pre[t+1,i] = f(lmd[i],data_pre[t+1,i])
    return data_pre

#模糊认知图递推过程训练用   
def fcmm(e,lmd,data_front):
    c_len = len(data_front)
    data_pre = np.dot(data_front,e)
    for i in range(c_len):
        data_pre[i] = f(lmd[i],data_pre[i])
    
    return data_pre

    #%% 
if __name__ == "__main__":
    data = pd.read_csv('dataProcess.csv',index_col = 0)
    data = np.array(data)
    train_time = 504               #设置滑动窗口
    start = 1
    hour = 100   
    concept = data.shape[1]
    
    data_test = data[start+train_time:start+train_time+hour,:]     #3天做测试
    data_pre = np.zeros((data_test.shape[0],data_test.shape[1]))
    data_train = data[start:train_time+start,:]
    e,lmd,error_time = jayaTrain(data[start-1,:],data_train,train_time,concept,npop = 8)
    
    for i in range(0,hour):
        #data_train = data[i+start:i+train_time+start,:]
        data_real = data[i+start+train_time-1,:]
        #e,lmd,error_time = jayaTrain(data[i+start-1,:],data_train,train_time,concept,npop = 8)
        data_pre[i] = fcmm(e,lmd,data_real)

    #print(errorLp(2,data_pre,data_test))
    drawAll(data_pre,data_test)
    #result(data_pre,data_test,'jaya_py')
    plt.plot(ed)
    plt.show()