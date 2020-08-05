import numpy as np
import pandas as pd
import math
def topsiscode(data, matrix, sign):
    #dataset = pd.read_csv('data')    
    dataset = data
    x = dataset.shape[0]
    y = dataset.shape[1]

    summ = sum(matrix)
    
    s = []
    for j in range(y):
        s.append(0)
        for i in range(x):
            s[j] = s[j]+(dataset[dataset.columns[j]].iloc[i])**2
        s[j]=np.sqrt(s[j])

    for j in range(y):
        for i in range(x):
            mat2=dataset[dataset.columns[j]].iloc[i]*matrix[j]/summ/s[j]
            #dataset.set_value(i, dataset.columns[j], mat2)
            dataset[dataset.columns[j]].iloc[i] = mat2
        vmax = []
        vmin =[]

    for i in  range(y):
        if sign[i]=='+':
            vmax.append(dataset[dataset.columns[i]].max())
            vmin.append(dataset[dataset.columns[i]].min())
        elif sign[i]=='-':
            vmax.append(dataset[dataset.columns[i]].min())
            vmin.append(dataset[dataset.columns[i]].max())
    sp = []
    sn = []

    for i in range(x):      
        sp.append(0)
        sn.append(0)
        for j in range(y):
            sp[i]=sp[i]+(dataset[dataset.columns[j]].iloc[i]-vmax[j])**2
            sn[i]=sn[i]+(dataset[dataset.columns[j]].iloc[i]-vmin[j])**2
        sp[i]=np.sqrt(sp[i])
        sn[i]=np.sqrt(sn[i])

    final = []
    for i in range(x):
        final.append(0)
        final[i]=sn[i]/(sn[i]+sp[i])  
     
    final_date = []
    for i in range(int(x/24)):
        final_date.append(0)
        for j in range(24):
            final_date[i] += final[i*24+j]
        final_date[i] = final_date[i] / 24
        
    rs = pd.date_range(data.index[0],data.index[-1],freq='d')        
    final_date = pd.DataFrame(final_date,index=rs,columns=['rank'])
    f = final_date.sort_values('rank',inplace=False,ascending=False)
    return f,final_date


def seriesWeight(t_k,t_c):
    s_w = [0]*t_k
    t_f = 2*t_c-1
    mu = t_c
    sigma_2 = 0
    for i in range(1,t_f+1):
        sigma_2 += (i-mu)**2
    sigma_2 = sigma_2/t_f
    temp = 0
    for i in range(1,t_k+1):
        for j in range(1,t_k+1):
            temp += math.exp((j-mu)**2/(2*sigma_2))
        s_w[i-1] = math.exp((i-mu)**2/(2+sigma_2))/temp
    return s_w

if __name__ == "__main__":
    data = pd.read_csv('dataProcess.csv',index_col = 0)
    matrix = [0.2,0.1,0.1,0.1,0.3,0.2]
    sign = ['-','-','-','-','-','-']
    rank ,final_date =topsiscode(data,matrix,sign)
    