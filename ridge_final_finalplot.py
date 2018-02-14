# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:14:35 2017

@author: NIRMAL ELAMON
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:03:19 2017

@author: NIRMAL ELAMON
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:40:26 2017

@author: NIRMAL ELAMON
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import tree
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import warnings
from math import sqrt
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)



#NUM_THREADS =400
#sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))

#path = 'C:\\Users\\adity\\Documents\\NCSU\\2 ECE765\\Proj1\\Part2\\DataFeatures4.1.CSV'
col_arr = [0]*54
for i in range(0,54):
    col_arr[i] = i

 
path1 ='F:\masters\ece 765\project 1\Trainning Data\Day1\Post\DataFeatures1.csv'
path2 ='F:\masters\ece 765\project 1\Trainning Data\Day2\Post1\DataFeatures2.csv' 
path3 ='F:\masters\ece 765\project 1\Trainning Data\Day2\Post2\DataFeatures.csv' 
path4 ='F:\masters\ece 765\project 1\Trainning Data\Day3\Post\DataFeatures4.csv'
path5 ='F:\masters\ece 765\project 1\Trainning Data\Day4\Post\DataFeatures.csv'

df=(pd.read_csv(path1 ,header = None, usecols = col_arr))
df1=(pd.read_csv(path2 ,header = None, usecols = col_arr))
df2=(pd.read_csv(path3 ,header = None, usecols = col_arr))
df3=(pd.read_csv(path4 ,header = None, usecols = col_arr))
df4=(pd.read_csv(path5 ,header = None, usecols = col_arr))



df=np.array(df)
df1=np.array(df1)
df2=np.array(df2)
df3=np.array(df3)
df4=np.array(df4)

df5=np.concatenate((df,df1))
df6=np.concatenate((df5,df2))
df7=np.concatenate((df6,df3))
df8=np.concatenate((df7,df4))

df8=pd.DataFrame(df8)
df8 = df8.fillna(0)

f = [1,2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19,20,24,25,26,27,28,29,31,32,33,34,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52]
#df = pd.read_csv(path,header = None, usecols = col_arr)
#df = df.fillna(0)
#print('1')
y = df8[53]
x= df8[f]
#x = df8[df8.columns[1:53]]
#R2=[0]*44
#xR2 = [0]*44
#defaultRMSE=[]


yHat1=[]
y_Train=[]
X_Train=[]
pca_components = 42
n_components = min(x.shape[0],pca_components)  

#PCA  
pca = PCA(n_components)
#Fits the feature tp PCA and returns the features reduced in dimentionality
X_pca = pca.fit_transform(x)
X = np.array(X_pca)
y = np.array(y)


len_CV=int(0.8*len(X))
len_Test=len(X)-len_CV
x_Train=[]
y_Train=[]

for i in range(0,len_CV):
    x_Train.append(X[i])
    y_Train.append(y[i])
x_Train=np.array(X)
y1_Train=np.array(y)
x_Test=[]
y_Test=[]
for i in range(len_CV,len(X)):
    x_Test.append(X[i])
    y_Test.append(y[i])    
x_Test=np.array(x_Test)
y1_Test=np.array(y_Test)

x_Train=np.array(x_Train)
y1_Train=np.array(y1_Train)


x_Test=np.array(x_Test)
y1_Test=np.array(y1_Test)

# K-Fold Cross Validation
TR=0
# Number of Folds, CV_data = Total_Smaples/Fold
Fold = 10        
kf = KFold(n_splits = Fold)
i=0
o=0
R=0
RMSE = 0
Ro=0

clf =  Ridge(alpha=100)

ccc=1
bbb=1
r2folds=[]
msefolds=[]
rmsfolds=[]
for train, test in kf.split(X,y):
    #print("%s - %s" %(train, test))
    
    #Training Data
    X_train = x_Train[train][:]
    y_train = y1_Train[train][:]
    
    
    #y_temp = np.array([y_train])
    #con = np.concatenate((X_train,y_temp.T),axis=1)
    #np.random.shuffle(con)
    #y_train = con[:,pca_components]
    #X_train = con[:,0:pca_components]
    #print('30')
    # CV data
    X_CV = x_Train[test][:]
    y_CV = y1_Train[test][:]
    
    #Ridge Reg Fitting
    
    clf.fit(X_train, y_train)
    i=i+1
    #print('InterationNo.',i)
    #print('40')
    #  Length of CV data
    len_CV = test.shape[0]
    len_train = train.shape[0]
    #yHat = [0]*len_CV
    
    
    #while i < len_CV:
        #XXX = np.array(X_CV)
    #while i < Fold:
    #yHat1 = clf.predict(X_CV)
    '''
    r2 = r2_score(y_CV,yHat1,multioutput='variance_weighted')
    r2folds.append(r2)       
    mse=mean_squared_error(y_CV, yHat1)
    msefolds.append(mse)
    rms = sqrt(mse)
    rmsfolds.append(rms)
    '''
    
    yHat1 = np.array(yHat1)
    if bbb ==1:
        yHat = yHat1
        bbb = 20
    else:
        yHat = np.append(yHat, yHat1)
    
    if ccc == 1:
        yCV1 = np.array(y_CV)
        ccc = 20
    else:
        yCV1 = np.append(yCV1, y_CV)

yhattest = clf.predict(x_Test)
xplot = [0]*len(y_Test)
for i in range (0,len(y_Test)):
    xplot[i] = i
plt.plot(xplot,yhattest)
plt.plot(xplot,y_Test)
e1mse = np.mean(np.square(yhattest-y_Test))
e1rmse = np.sqrt(np.mean(np.square(yhattest-y_Test)))
R2 = r2_score(yhattest,y_Test,multioutput='variance_weighted')
'''
    mean_CV = np.mean(yCV1)
    print('Mean',mean_CV)
    
    
    print('LengthCVData', len_CV)
    print('LengthTrainData', len_train)
    
    len_Total =  len_CV + len_train
    print('LengthTotalData', len_Total)
    i=0
    xplot = [0]*len_Total
    meanplot = [0]*len_Total
    for i in range (0,len_Total):
        xplot[i] = i
        meanplot[i] = mean_CV
    
    
    
    #plt.plot(xplot,yHat) # Blue
    #plt.plot(xplot,yCV1) # Orange
    #plt.plot(xplot,meanplot)#Green
    
    r2 = r2_score(yCV1,yHat,multioutput='variance_weighted')
    print('R2',r2)
    R2[z]=r2
    print('R2',R2[z])
    
    
    
    
    e1 = np.sqrt(np.mean(np.square(yCV1-mean_CV)))
    print('ErrorTrue',e1)
    e2 = np.sqrt(np.mean(np.square(yHat-mean_CV)))
    print('ErrorPred',e2)
    
    ef = np.abs(e1-e2)
    print('Error', ef)
    
    
    
    
    e_rmse = np.sqrt(e_mse)
    print('MSE = ', e_mse)
    print('RMSE = ', e_rmse)

'''
