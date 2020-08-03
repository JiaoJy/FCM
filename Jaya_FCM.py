# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import copy

import dataPreprocessM as dprc


def f(lmd,c):
    y = 1 / (1 + np.exp(-lmd*c))
    return y

def h(err):
    y = 1 / (2 * err + 1)
    return y

def errorLp(p,data_pre,data_real):
    dist = np.linalg.norm(data_pre-data_real,ord=p)
    return dist
  

#返回初始化权重参数   NxN 
def initialize_parameters_he(npop,N):
    parameters = np.zeros((npop,N*N))
    for i in range(npop):
        parameter = np.random.randn(N*(N-1)) * np.sqrt(2.0 / (N*(N-1)))
        lmd = np.random.uniform(1,5,N)
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
    #print(X.shape)
    for n in range(npop):
        for i in range(N*N):
            X[n,i] = X[0,i] + np.random.random()*(X[npop-1,i]-X[0,i])
    fitness = np.zeros(npop)
    error = np.zeros(npop)
    worst= 0
    best = 0
    for num in range(500):
        for n in range(npop):
            e,lmd = rshape(X[n,:],N)
            print(e,lmd)
            c_pre = fcm(e,lmd,c_data,time)
            fitness[n] = h(errorLp(2,c_pre,c_real)) 
            error[n] = errorLp(2,c_pre,c_real)
        worst = fitness.tolist().index(min(fitness))
        best = fitness.tolist().index(max(fitness))
        print(worst,best)
    
        Xx = np.zeros((npop,N*N))
        for n in range(npop):
            Xx[n,:] = X[n,:] + (X[best,:]-np.abs(X[n,:]))*np.random.random()-(X[worst,:]-np.abs(X[n,:]))*np.random.random()
            for i in range(N*N):
                if Xx[n,i] < Xx[0,i]:
                    Xx[n,i] = Xx[0,i]
                if Xx[n,i] > Xx[npop-1,i]:
                    Xx[n,i] = Xx[npop-1,i]
            e,lmd = rshape(X[n,:],N)
            c_pre = fcm(e,lmd,c_data,time)
            error2 = errorLp(2,c_pre,c_real)
            if error2 <= error[n]:
                X[n,:] = Xx[n,:]
    e,lmd = rshape(X[best,:],N)
    return e,lmd
       
def fcm(e,lmd,data,time):
    data_pre = np.zeros((time+1,len(data)))
    data_pre[0] = data
    for t in range(0,time):
        for i in range(len(data)):
            for j in range(len(data)):
                if i is not j :
                    data_pre[t+1,i] += e[j,i]*data_pre[t,j]
            data_pre[t+1,i] += data_pre[t,i]
            data_pre[t+1,i] = f(lmd[i],data_pre[t+1,i])
    return data_pre
          

if __name__ == "__main__":
    data = dprc.dataPreprocess("数据20171205\数据\沧州渤海临港产业园","XH8082015110300850.csv")
    data = np.array(data)
    data_train = data[20:500,:]
    data_test = data[500:600,:]
    e,lmd = jayaTrain(data_train[20,:],data_train,479,8,4)
    data_pre = fcm(e,lmd,data_test[0,:],99)
    print(errorLp(2,data_pre,data_test))
    
    