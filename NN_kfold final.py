import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import math
#from keras.models import load_model
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#NUM_THREADS =6000
#sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))


warnings.filterwarnings("ignore", category=DeprecationWarning)


'# layer1 = 130, error= 5.5580386248685887, R2 = -0.49325209577477036#'
'# layer1 = 200, error= 5.7870608452713217, R2 = -0.57045808696726019#'
'# layer1 = 40,  error= 5.1991030120977602, R2 = -0.47089198598086796#'

def newModel(num_features):
    rms = RMSprop()
    model = Sequential()
    model.add(Dense(40, input_dim=num_features, kernel_initializer='lecun_uniform',activation='relu' ))
    model.add(Dense(1, kernel_initializer='lecun_uniform'))
    model.compile(loss='mse', optimizer=rms,metrics=['mse'])
    return model
 

def train(X_train_NN,y_train_NN,model,len_train):
    len_train = int(len_train)
    for i in range(0,len_train):
        f = [X_train_NN[i]]
        g = np.array(f)
        model.fit(g,np.array([y_train_NN[i]]),epochs=1,batch_size=100,verbose=0)
    model.save('my_model.h5')
    return model


def perform_PCA(X,pca_components):
    pca = decomposition.PCA(n_components=pca_components)
    pca.fit(X)
    X_PCA = pca.transform(X)
    return X_PCA


def preprocess(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_PP = scaler.transform(X)
    return X_PP
    


path1 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day1/Post/DataFeatures.csv'
path2 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day2/Post1/DataFeatures.csv'
path3 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day2/Post2/DataFeatures.csv'
path4 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day3/Post/DataFeatures.csv'
path5 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day4/Post/DataFeatures.csv'

col_arr = [0]*54
for i in range(0,54):
     col_arr[i] = i
 
df1 = pd.read_csv(path1 ,header = None, usecols = col_arr)
df2 = pd.read_csv(path2 ,header = None, usecols = col_arr)
df3 = pd.read_csv(path3 ,header = None, usecols = col_arr)
df4 = pd.read_csv(path4 ,header = None, usecols = col_arr)
df5 = pd.read_csv(path5 ,header = None, usecols = col_arr)

res = [df1,df2,df3,df4,df5]
df = pd.concat(res)
df = df.reset_index(drop=True)  

df = df.fillna(0)
Total = len(df)


num_features = 44
'# Write column index in feature_arr of only those #' 
'# columns which you want to consider as your feature #'
feature_arr = [-1]*num_features
for i in range(1,num_features+1):
    feature_arr[i-1] = i
    
feature_arr = [1,2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19,20,24,25,26,27,28,29,31,32,33,34,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52]

pca_components = 20
print('PCA',pca_components)
     
y = df[53]
x = df[feature_arr]
x = np.array(x)

#Total = 1000
K = 10
Total = (Total/K)*K
width = Total/K
Total = int(Total)
print (Total)

y_cv_r2 = []
y_hat_r2 = []


model = newModel(pca_components)
error_metric = [-1]*K
for k_index in range(0,K):
    print ("k_index = ",k_index)
    cv_start = k_index*width
    cv_end = cv_start + width
    train_start1 = 0
    train_end1 = cv_start
    train_start2 = cv_end
    train_end2 = Total
    
    if(cv_start == 0):
        train_end1=0
    if(cv_end == Total):
        train_start2 = Total
    len_train = Total*(K-1)/K
    len_CV = width
    
    print(train_start1,train_end1,cv_start,cv_end,train_start2,train_end2,k_index)
    
    X_train = []
    X_CV = []
    y_train = []
    y_CV = []    
    
    train_start1=int(train_start1)
    train_end1=int(train_end1)
    train_start2=int(train_start2)
    train_end2=int(train_end2)
    cv_start=int(cv_start)
    cv_end=int(cv_end)
    
   
    
    for i in range (train_start1,train_end1):
        X_train.append(x[i])
        y_train.append(y[i])
    for i in range (train_start2,train_end2):
        X_train.append(x[i])
        y_train.append(y[i])    
    for i in range (cv_start,cv_end):    
        X_CV.append(x[i])
        y_CV.append(y[i])
    
    print ("Performing PCA")
    X_train_NN = perform_PCA(X_train,pca_components)
    X_CV_NN = perform_PCA(X_CV,pca_components)
 
     
    print ("Randomizing inputs")
    y_temp = np.array([y_train])
    con = np.concatenate((X_train_NN,y_temp.T),axis=1)
    np.random.shuffle(con)
    y_train = con[:,pca_components]
    X_train_NN = con[:,0:pca_components]
    y_train_NN = y_train
    y_CV_NN = y_CV
 
    
    print ("Preprocessing") 
    X_train_NN = preprocess(X_train_NN)
    X_CV_NN = preprocess(X_CV_NN)

    print ("Training")
    model = train(X_train_NN,y_train_NN,model,len_train)
    
    print ("Testing")
    yHat = []
    for j in range(cv_start,cv_end):
        f = [X_CV_NN[j-cv_start]]
        g = np.array(f)
        y_out = model.predict(g)
        yHat.append(y_out[0][0])

    mean = np.sum(y_CV)/len_CV
    
    y_cv_r2.extend(y_CV)
    y_hat_r2.extend(yHat)
    
    diff_CV = y_CV - mean
    diff_Hat = yHat - mean
    CV_sq = np.sum(diff_CV*diff_CV)/len_CV
    Hat_sq = np.sum(diff_Hat*diff_Hat)/len_CV
    error_sq = CV_sq-Hat_sq
    error = math.sqrt(abs(error_sq))
    print ("Error = ",error)
    error_metric[k_index] = error
    
    
r2val = r2_score(y_cv_r2,y_hat_r2,multioutput='variance_weighted')
error_metric = np.array(error_metric)
total_error = np.sum(error_metric)/K
print ("Total Error = ",total_error)
print ("R2 = ",r2val)

#K_arr = [-1]*K
#for i in range(0,K):
#    K_arr[i] = i
    
#plt.plot(K_arr,error_metric)    



