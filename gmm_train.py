import pandas as pd
import sklearn.mixture as mix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

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


num_features = 52
pca_components = 20
day = 5
num_components = 4


path1 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day1/Post/DataFeatures.csv'
path2 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day2/Post1/DataFeatures.csv'
path3 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day2/Post2/DataFeatures.csv'
path4 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day3/Post/DataFeatures.csv'
path5 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day4/Post/DataFeatures.csv'
path = '/home/suraj/NCSU/765/Project1/P4/gmm/trained_data/'
path = path + 'day'+str(day)+'_'+str(num_components)+'.csv'

col_arr = [i for i in range(0,54)]  
feature_arr = [i for i in range(1,num_features+1)]

df1 = pd.read_csv(path1 ,header = None, usecols = col_arr)
df2 = pd.read_csv(path2 ,header = None, usecols = col_arr)
df3 = pd.read_csv(path3 ,header = None, usecols = col_arr)
df4 = pd.read_csv(path4 ,header = None, usecols = col_arr)
df5 = pd.read_csv(path5 ,header = None, usecols = col_arr)

if day == 1:
    df = df1
elif day == 2:
    df = df2    
elif day == 3:
    df = df3
elif day == 4:
    df = df4
elif day == 5:
    df = df5

df = df.fillna(0)

X = df[feature_arr]
Y = df[53]

Total = len(df)
len_train = Total
X = np.array(X)
Y = np.array(Y)

X = perform_PCA(X,pca_components)
X = preprocess(X)

X_train = []
Y_train = []
X_CV = []
Y_CV = []

for i in range(0,len_train):
    X_train.append(X[i])
    Y_train.append(Y[i])
        

X_train = np.array(X_train)
Y_train = np.array(Y_train)


print "############################"

model1 = mix.GaussianMixture(n_components=num_components, 
                            covariance_type="full", 
                            n_init=100, 
                            random_state=7,
                            verbose=1).fit(X_train)

hs_train = model1.predict(X_train)


hs_train = np.array(hs_train)
Y_train = np.array(Y_train)

df_hs = pd.DataFrame(hs_train)
df_ytrain = pd.DataFrame(Y_train)

res = [df_hs, df_ytrain]
d = pd.concat(res, axis = 1)
d = d.reset_index(drop=True)
  
d.to_csv(path, header=None,sep=',')






