__author__ = '15275'
import numpy as np
def sgdForFM(datamartrix,classlabel,factor,iter,alpha):
    m,n=np.shape(datamartix);
    w_0=0;
    w=np.zeros((n,1));
    v=np.random.normal(0,0.1,size=(n,factor));
    for mm in range(iter):
        for number in range(m):
            data_number=datamartrix[number].reshape(1,n);
            inter1=np.matmul(data_number,v);
            inter2=np.matmul(np.multiply(data_number,data_number),np.multiply(v,v))
            interaction=np.sum(inter1*inter1-inter2)/2
            prediction=w_0+np.matmul(data_number,w)+interaction;
            loss=((classlabel[number]-prediction[0,0])**2)/2;
            print(loss)
            partial_loss=(classlabel[number]-prediction[0,0])
            #更新w-0
            w_0=w_0+alpha*partial_loss
            for feather_number in range(n):
                 #更新w_i
                w[feather_number,0]=w[feather_number,0]+alpha*partial_loss*data_number[0,feather_number]
                #更新v[feather_number,factor_number]
                for factor_number in range(factor):
                    #对V_if偏导数的计算
                    patial_vif=data_number[0,feather_number]*inter1[0,factor_number]-v[feather_number,factor_number]*(data_number[0,feather_number]**2);
                    v[feather_number,factor_number]=v[feather_number,factor_number]+alpha*partial_loss*patial_vif;
datamartix=np.array([[1,0,0],[0,0,0],[4,0,0]])
label=np.array([[1],[0],[4]])
sgdForFM(datamartix,label,2,100,0.01)