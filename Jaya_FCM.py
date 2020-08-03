# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

import dataPreprocessM as dprc


def f(lmd,c):
    y = 1 / (1 + np.exp(-lmd*c))
    return y



def errorLp(p,data_pre,data_real):
    dist = np.linalg.norm(data_pre-data_real,ord=p)
    return dist
    

def jayaTrain(data_train_pre,data_train_real):
    
    


def fcm(t,e,lmd):
    

if "__main__":
    
    