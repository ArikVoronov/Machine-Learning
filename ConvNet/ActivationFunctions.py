import numpy as np
# %% Activation functions
def Softmax(z,derive):
    e_sum = np.sum(np.exp(z),axis=0)
    a = np.exp(z)/e_sum
    if derive == 0:
        y = a
    elif derive == 1:
        y = a*(1-a)
    return y

def ReLU(z,derive):
    if derive == 0:
        y = z*(z>0) 
    elif derive == 1:
        y = 1*(z>0) 
    return y
    
def ReLU2(z,derive):
    if derive == 0:
        y = z*(z<=0)*0.1 + z*(z>0) 
    elif derive == 1:
        y = 0.1*(z<=0) + (z>0) 
    return y
