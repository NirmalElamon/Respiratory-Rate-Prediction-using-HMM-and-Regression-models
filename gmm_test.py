import pandas as pd
import sklearn.mixture as mix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

num_features = 52
pca_components = 20
day_CV = 5
num_components = 4
start = 16000
len_CV = 16000

path1 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day1/Post/DataFeatures.csv'
path2 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day2/Post1/DataFeatures.csv'
path3 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day2/Post2/DataFeatures.csv'
path4 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day3/Post/DataFeatures.csv'
path5 = '/home/suraj/NCSU/765/Project1/Trainning Data/Day4/Post/DataFeatures.csv'


col_arr = [i for i in range(0,54)]  
feature_arr = [i for i in range(1,num_features+1)]

df1 = pd.read_csv(path1 ,header = None, usecols = col_arr)
df2 = pd.read_csv(path2 ,header = None, usecols = col_arr)
df3 = pd.read_csv(path3 ,header = None, usecols = col_arr)
df4 = pd.read_csv(path4 ,header = None, usecols = col_arr)
df5 = pd.read_csv(path5 ,header = None, usecols = col_arr)



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

def get_state_means(sr):
    value_sum_arr = [0]*num_components
    value_count_arr = [0]*num_components
    
    for state_num in range(0,num_components):
        for j in range(0,len(sr)):
            if(sr[j,0] == state_num):
                value_sum_arr[state_num] += sr[j,1]
                value_count_arr[state_num] += 1

    value_sum_arr = np.array(value_sum_arr)
    value_count_arr = np.array(value_count_arr)
    state_mean = value_sum_arr/value_count_arr                        
    return state_mean

def get_yHat(hs_CV,sr,state_mean):
    len_CV = len(hs_CV)
    yHat = [0]*len_CV
    for i in range(0,4):
        yHat[i] =  state_mean[int(sr[hs_CV[i],0])]
    
    for i in range(4,len_CV):
        print "iteration",i
        count0,count1,count2,count3,count4 = 0,0,0,0,0
        val0,val1,val2,val3,val4 = 0,0,0,0,0 
        
        for j in range(4,len(sr)):
            if(sr[j,0] == hs_CV[i]):
                count0 += 1
                val0 = val0 + sr[j,1]
                if(sr[j-1,0] == hs_CV[i-1]):
                    count1 += 1
                    val1 = val1 + sr[j-1,1]
                    if(sr[j-2,0] == hs_CV[i-2]):
                        count2 += 1
                        val2 = val2 + sr[j-2,1]
                        if(sr[j-3,0] == hs_CV[i-3]):
                            count3 += 1
                            val3 = val3 + sr[j-3,1]
                            if(sr[j-4,0] == hs_CV[i-4]):
                                count4 += 1
                                val4 = val4 + sr[j-4,1]
        
        if(count4 != 0):
            yHat[i] = val4/count4
        elif(count3 != 0):
            yHat[i] = val3/count3                
        elif(count2 != 0):
            yHat[i] = val2/count2
        elif(count1 != 0):
            yHat[i] = val1/count1
        elif(count0 != 0):
            yHat[i] = val0/count0
            
    return yHat                    

def get_sr(day_model):
    path = '/home/suraj/NCSU/765/Project1/P4/gmm/trained_data/'    
    path = path + 'day'+str(day_model)+'_'+str(num_components)+'.csv'    
    df_model = pd.read_csv(path ,header = None, usecols = [1,2])
    sr = np.array(df_model)
    return sr



if day_CV == 1:
    df = df1
elif day_CV == 2:
    df = df2    
elif day_CV == 3:
    df = df3
elif day_CV == 4:
    df = df4
elif day_CV == 5:
    df = df5

df = df.fillna(0)

X = df[feature_arr]
Y = df[53]
X = np.array(X)
Y = np.array(Y)

X = perform_PCA(X,pca_components)
X = preprocess(X)

X_CV = []
Y_CV = []


for i in range(start,start+len_CV):
    X_CV.append(X[i])
    Y_CV.append(Y[i])
        
X_CV = np.array(X_CV)
Y_CV = np.array(Y_CV)


print "GMM    ############################"

model = mix.GaussianMixture(n_components=num_components, 
                            covariance_type="full", 
                            n_init=100, 
                            random_state=7,
                            verbose=1).fit(X_CV)

hs_CV = model.predict(X_CV)
hs_CV = np.array(hs_CV)

sr1 = get_sr(day_model = 1)
sr2 = get_sr(day_model = 2)
sr3 = get_sr(day_model = 3)
sr4 = get_sr(day_model = 4)

sr = np.concatenate((sr1,sr2,sr3,sr4), axis = 0)
state_mean = get_state_means(sr)
yHat = get_yHat(hs_CV,sr,state_mean)

yHat = np.array(yHat)
Y_CV = np.array(Y_CV)

dst = distance.euclidean(Y_CV,yHat)
rmse = dst/np.sqrt(len_CV)
print "RMSE = ",rmse

t = [i for i in range(0,len_CV)]
plt.plot(t,Y_CV,t,yHat)


