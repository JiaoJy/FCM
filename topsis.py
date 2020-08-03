import dataPreprocessM as dprc
import numpy as np
import pandas as pd
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

    final = pd.DataFrame(final,index=data.index,columns=['rank'])
    f = final.sort_values('rank',inplace=False,ascending=False)
    return(f)


def seriesWeight(t_k,t_c):
    s_w = [0]*t_k
    mu = (1+t_k)/2
    for i in range(1,t_k+1):
        sigma_2 += (i-mu)**2
    sigma_2 = 1/t_k * sigma
    temp = 0
    for i in range(1,k+1):
        for j in range(1,t_k+1):
            temp += exp((j-mu)**2/(2*sigma_2))
        s_w[i] = exp((i-mu)**2/(2+sigma_2))/temp
    return s_w

if __name__ == "__main__":
    data = dprc.dataPreprocess("数据20171205\数据\沧州渤海临港产业园","XH8082015110300850.csv")
    print(data)
    matrix = [0.2,0.1,0.05,0.1,0.2,0.2,0.1,0.05]
    sign = ['-','+','-','+','-','-','+','-']
    print(topsiscode(data,matrix,sign))
    