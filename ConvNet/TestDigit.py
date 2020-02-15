from PIL import Image
import pandas as pd
import numpy as np
import pickle

import ConvNet
import ActivationFunctions as af

def one_hot_output(y):
    y_vector = np.zeros([y.max()+1,len(y.T)])
    for i,out in enumerate(y.T):
        y_vector[out,i] = 1
    return y_vector
def vectorize_image(name):
    # Turn 3 channels 2D image into 1D vector
    img = Image.open(name)
    arr = np.array(img)
    n = arr.shape[0]
    m = arr.shape[1]
    pixel_matrix = arr[:, :, 1].reshape(n*m,1) # green value for a pixel
    return pixel_matrix
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
b_size = 256 # Batch size
clf = ConvNet.network(epochs,tolerance,actuators,layer_parameters,layer_types,alpha,beta1,beta2,epsilon,gamma,lam)

#%% Predict

df = pd.read_csv('digits_test.csv')
test_samples = 2000
print('Predicting using {} test samples'.format(test_samples))
x_test =  np.array(df.drop('label',1).loc[:].values).astype(int).T
x_test = x_test[:,-test_samples:]
x_test = x_test.reshape(28,28,1,-1)

# Output - one hot - y[ouputs,samples]
output_data_test = np.array(df[['label']].loc[:].values).astype(int).T
y_test = one_hot_output(output_data_test[:,-test_samples:])


##clf.x = x_test
clf.y = y_test

clf.x,clf.x_mean,clf.x_std = clf.Normalize(x_test)
clf.m = x_test.shape[-1] #number of samples
clf.OrganizeLayers()
wv0,bv0 = clf.initialize()
# %% Pickle Network Parameters
with open('pickled.dat', 'rb') as f:
    [w_saved, b_saved] = pickle.load(f)
clf.b = b_saved
clf.w = w_saved

a_pred,y_pred = clf.predict(x_test)

y_error = np.sum((y_test-y_pred)**2)/y_test.shape[1]*100
print('Total error percentage:',y_error, '%')


##
# %% Test
mat = vectorize_image('TestDigit.png')
mat = mat.reshape(28,28,1,1)
_,y_test = clf.predict(mat)
print('Prediction digit:',np.argmax(y_test))
