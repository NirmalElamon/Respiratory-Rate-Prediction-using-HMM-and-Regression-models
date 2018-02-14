import numpy as np
import pandas as pd
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn import decomposition
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=DeprecationWarning)



def loadModel():
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model1 = model_from_json(loaded_model_json)
    # load weights into new model
    model1.load_weights("model.h5")
    print("Loaded model from disk")
    return model1


    

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
    


path = '/home/suraj/NCSU/765/Project1/P2/Final/testdata.csv'

col_arr = [0]*53
for i in range(0,53):
     col_arr[i] = i
 
df = pd.read_csv(path ,header = None, usecols = col_arr)


df = df.fillna(0)
length = len(df)


num_features = 44
feature_arr = [-1]*num_features
for i in range(1,num_features+1):
    feature_arr[i-1] = i
    
feature_arr = [1,2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19,20,24,25,26,27,28,29,31,32,33,34,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52]

pca_components = 20
print('PCA',pca_components)
     
X_test = df[feature_arr]
X_test = np.array(X_test)
    
    
print ("Performing PCA")
X_test_NN = perform_PCA(X_test,pca_components)
 
         
print ("Preprocessing") 
X_test_NN = preprocess(X_test_NN)


rms = RMSprop()
model = loadModel()
model.compile(loss='mse', optimizer=rms,metrics=['mse'])

yHat = np.zeros(length)

print "Testing"
for i in range(0,length):
     f = [X_test_NN[i]]
     g = np.array(f)
     yHat[i] = model.predict(g)

yHat = np.array(yHat)    

predfile = open("yPred.txt","w")
for i in range(length):
    line = str(int(yHat[i])) + ",\n"
    predfile.write(line)
predfile.close()        

df2 = pd.DataFrame(yHat)

