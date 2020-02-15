import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
import numpy as np
import pickle

import ConvNet
import ActivationFunctions as af


def vectorize_image(name):
    # Turn 3 channels 2D image into 1D vector
    img = Image.open(name)
    arr = np.array(img)
    n = arr.shape[0]
    m = arr.shape[1]
    pixel_matrix = arr[:, :, 1].reshape(n*m,1) # green value for a pixel
    return pixel_matrix

def one_hot_output(y):
    y_vector = np.zeros([y.max()+1,len(y.T)])
    for i,out in enumerate(y.T):
        y_vector[out,i] = 1
    return y_vector


#%% Main

# Import data as pandas object
df = pd.read_csv('digits_full.csv')

## 
#%% Organize input for ConvNet
# Shuffle randomization seed
np.random.seed(1)

# Input x[inputs,samples]
samples = 40e3
x_train = np.array(df.drop('label',1).loc[:(samples-1)].values).astype(int).T
x_train = x_train.reshape(28,28,1,-1)

# Output - one hot - y[ouputs,samples]
output_data =  np.array(df[['label']].loc[:(samples-1)].values).astype(int).T
y_train = one_hot_output(output_data)

##x_train = np.random.rand(28,28,1,int(samples))
##y_train = np.random.rand(10,int(samples))

## Hyper parameters, define Network
epochs = 10
tolerance = 1e-2

fh = 6
fw = 6
filters = 25
fcl_filters = 60
stride = 3
layer_parameters = [[fh,fw,filters,stride],
                    [3,3,filters,2],
                    [fcl_filters+10],
                    [fcl_filters]]
layer_types = ['conv','max_pool','fc','fc']
actuators = [[0] , af.ReLU2,None,af.ReLU2,af.ReLU2,af.Softmax]

alpha = 0.01 # Step size
beta1 = 0.9 # Step weighted average parameter
beta2 = 0.98 # Step normalization parameter
gamma = 0.001 # Decay mulptiplier at the end of training (epochs*batch_size)
epsilon = 1e-8 # Addition to denominator to prevent div by 0
lam = 1e-4 # Regularization parameter
b_size = 512 # Batch size
clf = ConvNet.network(epochs,tolerance,actuators,layer_parameters,layer_types,alpha,beta1,beta2,epsilon,gamma,lam)

# Train Neural Network
print('Training on {} samples'.format(samples))
clf.Train(x_train,y_train,batch_size = b_size,wv = None, bv = None)        

# %% Pickle Network Parameters
b_saved = clf.b
w_saved = clf.w

PIK = "pickled.dat"
data = [clf.w,clf.b]
with open(PIK, "wb") as f:
    pickle.dump(data, f)

##
#%% Predict

test_samples = 3000
print('Predicting using {} test samples'.format(test_samples))
x_test =  np.array(df.drop('label',1).loc[:].values).astype(int).T
x_test = x_test[:,-test_samples:]
x_test = x_test.reshape(28,28,1,-1)

# Output - one hot - y[ouputs,samples]
output_data_test = np.array(df[['label']].loc[:].values).astype(int).T
y_test = one_hot_output(output_data_test[:,-test_samples:])

a_pred,y_pred = clf.predict(x_test)

y_error = np.sum((y_test-y_pred)**2)/y_test.shape[1]*100
print('Total error percentage:',y_error, '%')


##
# %% Test
mat = vectorize_image('TestDigit.png')
mat = mat.reshape(28,28,1,1)
_,y_pred = clf.predict(mat)
print('Prediction digit:',np.argmax(y_pred))


