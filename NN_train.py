import numpy as np
import pandas as pd
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
#from keras.models import load_model
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import scipy

warnings.filterwarnings("ignore", category=DeprecationWarning)

'# layer1 = 130, error=  6.55356#'
'# layer1 = 200, error=  6.16480#'
'# layer1 = 40,  error=  5.07927#'

def saveModel(model):
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    

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
    print "model saved"
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
len_CV = 0
len_train = Total - len_CV

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
y = np.array(y)


X_train = []
y_train = []

X_CV = []
y_CV = []
    

print ("Randomizing")
y_temp = np.array([y])
con = np.concatenate((x,y_temp.T),axis=1)
np.random.shuffle(con)
y2 = con[:,44]
x2 = con[:,0:44]
 
    
for i in range (0,len_train):
    X_train.append(x2[i])
    y_train.append(y2[i])

#for i in range (len_train,Total):
 #   X_CV.append(x2[i])
 #   y_CV.append(y2[i])

#y_CV = np.array(y_CV)
    
print ("Performing PCA")
X_train_NN = perform_PCA(X_train,pca_components)
#X_CV_NN = perform_PCA(X_CV,pca_components)
 
     

print ("Preprocessing") 
X_train_NN = preprocess(X_train_NN)
#X_CV_NN = preprocess(X_CV_NN)


model = newModel(pca_components)



print ("Training")


for i in range(0,len_train):
    print float(100.0*i/len_train)
    model.fit(np.array([X_train_NN[i]]),np.array([y_train[i]]),epochs=1,batch_size=100,verbose=0)
    
saveModel(model)   

'''
yHat = np.zeros(len_CV)

for i in range(0,len_CV):
     f = [X_CV_NN[i]]
     g = np.array(f)
     yHat[i] = model.predict(g)

error = scipy.spatial.distance.euclidean(yHat, y_CV)/np.sqrt(len_CV)
print ("error = ",error)
'''
