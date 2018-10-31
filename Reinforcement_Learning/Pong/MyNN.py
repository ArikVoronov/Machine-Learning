import pandas as pd
import numpy as np
import time
import pickle

def vectorize_image(name):
    img = Image.open(name)
    arr = np.array(img)
    n = arr.shape[0]
    m = arr.shape[1]
    pixel_matrix = arr[:, :, 1].reshape(n*m,1) # green value for a pixel
    return pixel_matrix

def vector2image(vect):
    leng = int(np.sqrt(len(vect)))
    im = vect.reshape(leng,leng)
    return im


def one_hot_output(y):
    y_vector = np.zeros([y.max()+1,len(y.T)])
    for i,out in enumerate(y.T):
        y_vector[out,i] = 1
    return y_vector

def RMS(x,ax = None,kdims = False):
    y = np.sqrt(np.mean(x**2,axis = ax,keepdims = kdims ))
    return y

def conv_indices(x,f_parameters):
    # f_parameters [fh,fw,stride]
    
    x_rows = x.shape[0]
    x_cols = x.shape[1]
    
    f_rows = f_parameters[0]
    f_cols = f_parameters[1]
    stride = f_parameters[2]
    
    
    out_rows  = int(np.floor(( x_rows-f_rows)/stride+1))
    out_cols  = int(np.floor(( x_cols-f_cols)/stride+1))
    
    # The indexes for the first filter (top left of matrix)
    ind_base = (np.tile(np.arange(0,f_cols),(f_rows, 1))+x_cols*np.arange(0,f_rows).reshape(f_rows,1)).reshape(1,-1)
    
    # Tile the base indexes to rows and columns, representing the movement of the convolution
    ind_tile = np.tile(ind_base, (out_cols, 1))
    ind_tile += stride* np.arange(0,ind_tile.shape[0]).reshape(ind_tile.shape[0],1)
    ind_tile = np.tile(ind_tile, (out_rows, 1))
    
    rower = x_cols*(np.floor(np.arange(ind_tile.shape[0])/ind_tile.shape[0]*out_rows)).reshape(ind_tile.shape[0],1)
    
    ind_mat = ind_tile + rower.astype(int)*stride
    return ind_mat



def conv3d(x,f,stride):
    assert(f.shape[2]==x.shape[2]),'Inconsistent depths for filter and input'
    # x[x,y,channels,samples]
    # f[x,y,channels,filters]
    x_str = x.reshape(-1,x.shape[2],x.shape[3])
    f_str = f.reshape(-1,f.shape[2],f.shape[3])
    x_rows = x.shape[0]
    x_cols = x.shape[1]
    f_rows = f.shape[0]
    f_cols = f.shape[1]
    out_rows  = int(np.floor( (x_rows-f_rows)/stride + 1))
    out_cols  = int(np.floor( (x_cols-f_cols)/stride + 1))

    ind_mat = conv_indices(x,[f.shape[0],f.shape[1],stride])
    xb = x_str[ind_mat]
    xc = xb.swapaxes(2,1).reshape(xb.shape[0],xb.shape[1]*xb.shape[2],xb.shape[3])
    fc =f_str.swapaxes(0,1).reshape(f_str.shape[0]*f_str.shape[1],f.shape[3])
    conv_mat = np.dot(xc.swapaxes(1,2),fc).swapaxes(1,2).reshape(out_rows,out_cols,f.shape[3],x.shape[3])
    return conv_mat

def dz_calc(z,f,ind_mat,dz_next):
    f_str = f.reshape(-1,f.shape[2],f.shape[3])
    dz_n_str = dz_next.reshape(-1,dz_next.shape[2],dz_next.shape[3])
    

    chan_ind = np.arange(f.shape[2]).reshape(1,1,-1,1)
    filt_ind = np.arange(f.shape[3]).reshape(1,1,1,-1)
    dz_ind = np.arange(dz_n_str.shape[0]).reshape(-1,1,1,1)
    ind_m2 = np.tile(ind_mat[:,:,None,None],(1,1,f.shape[2],f.shape[3]))
    
    '''
     b - a sparse matrix, filled with filter members at convolution positions
     each col is a filter position, each row is a dz member,
     tiled over number of filters and channels
     
     the rows are multiplied by respective dz(next) members and then summed over
     summation INCLUDES filter index of dz(next)
    '''
    b_mat = np.zeros([dz_n_str.shape[0],z.shape[1]*z.shape[0],f.shape[2],f.shape[3]])
    b_mat[dz_ind,ind_m2,chan_ind,filt_ind]=f_str[None,:,:,:]
    
    # Method 1 :
    # stretch rows and filters into a long 
    b_mat2 = np.rollaxis(b_mat,0,4).reshape(b_mat.shape[1],b_mat.shape[2],-1)
    dz_n_str = dz_n_str.swapaxes(0,1).reshape(-1,dz_n_str.shape[2]).T
    dz_str = np.dot(dz_n_str,b_mat2.swapaxes(1,2))
    dz_str = np.rollaxis(dz_str,0,3)
    # Alternative (found to be less efficient)
#    dz_mat = dz_n_str[:,None,None,:]*b_mat[:,:,:,:,None]
#    dz_str = np.sum(np.sum(dz_mat, axis = 0),axis = -2)
    
    dz = dz_str.reshape(z.shape[0],z.shape[1],z.shape[2],z.shape[3])
    return dz

def dw_conv_indices(a0,dz_shape,filter_parameters):
    # filter_parameters [fh,fw,stride]
    fh = filter_parameters[0]
    fw = filter_parameters[1]
    stride = filter_parameters[2]
    x_rows = a0.shape[0]
    x_cols = a0.shape[1]
    
    f_rows = dz_shape[0]
    f_cols = dz_shape[1]
    
    
    out_rows  = fh
    out_cols  = fw
    
    # The indexes for the first filter (top left of matrix)
    ind_base = np.tile(np.arange(0,f_cols)*stride,(f_rows, 1))
    ind_base += x_cols*(np.arange(0,f_rows)*stride).reshape(f_rows,1)
    ind_base = ind_base.reshape(1,-1)
    
    # Tile the base indexes to rows and columns, representing the movement of the convolution
    ind_tile = np.tile(ind_base, (out_cols, 1))
    ind_tile += np.arange(0,ind_tile.shape[0]).reshape(ind_tile.shape[0],1)
    ind_tile = np.tile(ind_tile, (out_rows, 1))
    
    rower = x_cols*(np.floor(np.arange(ind_tile.shape[0])/ind_tile.shape[0]*out_rows)).reshape(ind_tile.shape[0],1)
    
    ind_mat = (ind_tile + rower.astype(int))
    
    return ind_mat
def dw_calc(a0,dz,filter_parameters):
    # filter_parameters [fh,fw,stride]
    fh = filter_parameters[0]
    fw = filter_parameters[1]

    a_str = a0.reshape(-1,a0.shape[2],a0.shape[3])
    dz_str = dz.reshape(-1,dz.shape[2],dz.shape[3])
    
    out_rows  = fh
    out_cols  = fw
    
    ind_mat = dw_conv_indices(a0,[dz.shape[0],dz.shape[1]],filter_parameters)
        #f_mat = np.tile(f_str,(bbb.shape[0],1))
    a_mat = a_str[ind_mat].swapaxes(2,1)
    ac = a_mat.swapaxes(2,3).reshape(a_mat.shape[0],a_mat.shape[1],a_mat.shape[3]*a_mat.shape[2])
    dzc = dz_str.T.swapaxes(0,1).reshape(dz_str.shape[1],dz_str.shape[0]*dz.shape[3])
    dzc = dzc.T
    dw = np.dot(ac,dzc)
    dw = dw.reshape(out_rows,out_cols,a0.shape[2],dz.shape[2])
    return dw

def dz_pool(z,x_max_ind,ind_mat,dz_next):
    # input z[L],x_max_ind[L+1],ind_mat[L+1],dz_next[L+1]
    out_rows = dz_next.shape[0]
    out_cols = dz_next.shape[1]
    b = np. zeros([z.shape[0]*z.shape[1],out_rows*out_cols,z.shape[2],z.shape[3]])
    dz_next_str = dz_next.reshape(-1,dz_next.shape[2],dz_next.shape[3])
    ind1 = np.arange(ind_mat.shape[0])
    ind2 = np.arange(z.shape[2])
    ind3 = np.arange(z.shape[3])
    indies = ind_mat[ind1[:,None,None],x_max_ind]
    b[indies[:,:,:,None],ind1[:,None,None,None],ind2[None,:,None,None],ind3[None,None,:,None]] = dz_next_str[:,:,:,None] 
    dz = np.sum(b,axis = 1).reshape(z.shape)
    return dz   
# %% Activation functions
def Softmax(z,derive):
    e_sum = np.sum(np.exp(z),axis=0)
    a = np.exp(z)/e_sum
    if derive == 0:
        y = a
    elif derive == 1:
        y = a*(1-a)
    return y
    
def ReLU2(z,derive):
    if derive == 0:
        y = z*(z<=0)*0.1 + z*(z>0) # get only the values larger than zero and normalize them
    elif derive == 1:
        z_fix = z+(z==0)*1e-3 # to make sure there are no zeroes for division, this is crazy
        y = z*(z<=0)/z_fix*0.1 + z*(z>0)/z_fix # get only the values larger than zero and normalize them
    return y
# %% Layer types

class max_pool_layer():
    def __init__(self,layer_sizes,layer_parameters):
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        self.lp = layer_parameters
        self.ls = layer_sizes
        self.stride = layer_parameters[1][-1]
    def initialize(self):
        w0 = 0
        b0 = 0
        return w0,b0
    def fp(self,w,b,a0):
        f_rows = self.lp[1][0]
        f_cols = self.lp[1][1]
        stride = self.stride
        
        out_rows = int( np.floor((a0.shape[0]-f_rows)/stride) )+1
        out_cols = int( np.floor((a0.shape[1]-f_cols)/stride) )+1
        a0_str = a0.reshape(-1,a0.shape[2],a0.shape[3])
        ind_mat = conv_indices(a0,[f_rows,f_cols,stride])
    
        a0_conv = a0_str[ind_mat]
        self.x_max_ind = np.argmax(np.abs(a0_conv),axis = 1)
        #a0_max = np.max(a0_conv,axis = 1)
        ind0 = np.arange(a0_conv.shape[0]).reshape(-1,1,1,1)
        ind2 = np.arange(a0_conv.shape[2]).reshape(1,1,-1,1)
        ind3 = np.arange(a0_conv.shape[3]).reshape(1,1,1,-1)
        
        a0_max = a0_conv[ind0, self.x_max_ind[:,None,:,:], ind2, ind3]
        a0_max = np.squeeze(a0_max,axis =1 )
        z = a0_max.reshape(out_rows,out_cols,a0_max.shape[1],a0_max.shape[2])
        a = z
        return z,a
    def bp(self,w,dz_next,z,a0,next_layer):
        #dz [L] : w[L+1],dz[L+1],z[L],a[L-1]
        if type(next_layer) == fully_connected_layer:
            # reshape to a column vector (cols = 1)
            z_str = z.reshape(-1,z.shape[-1])
            dz = np.dot(w.T,dz_next)
            dz = dz.reshape(z.shape)
        elif type(next_layer) == max_pool_layer:
            ind_mat = conv_indices(z,[self.lp[1][0],self.lp[1][1],self.stride]) 
            x_max_ind = next_layer.x_max_ind
            dz = dz_pool(z,x_max_ind,ind_mat,dz_next)
        elif type(next_layer) == conv_layer:
            ind_mat = conv_indices(z,[w.shape[0],w.shape[1],self.stride]) 
            dz= dz_calc(z,w,ind_mat,dz_next)
        db = 0
        dw = 0
        return dz,dw,db  
    
class conv_layer():
    def __init__(self,actuator,layer_sizes,layer_parameters):
        # layer_parameters is a list, [fh = filter_height,fw = filter_width ,channels,filters,stride]
        self.actuator = actuator
        self.lp = layer_parameters
        self.ls = layer_sizes
        self.stride = layer_parameters[1][-1]
    def initialize(self):
        ls = self.ls
        lp = self.lp
        zw = ls[1][0]
        zh = ls[1][1]
        fh = lp[1][0]
        fw = lp[1][1]
        filters = lp[1][2]
        channels = lp[0][2]
        f_total = fw*fh*channels*filters
        signs = (2*np.random.randint(0,2,size= f_total )-1).reshape([fh,fw,channels,filters] )
        var = np.sqrt(2/ls[1][2]) #  Not sure this is right for conv
        w0 = ((np.random.randint( 10,1e2, size=f_total )/1e2 ) ).reshape( [fh,fw,channels,filters ] )
        w0 = 1*var*signs*w0
        b0 = np.zeros([zw,zh,filters,1])
        return w0,b0
    def fp(self,w,b,a0):
        z = conv3d(a0,w,self.stride)
        z = z + b
        a = self.actuator(z,derive = 0)
        return z,a
    def bp(self,w,dz_next,z,a0,next_layer):
        if type(next_layer) == fully_connected_layer:
            # reshape to a column vector (cols = 1)
            z_str = z.reshape(-1,z.shape[-1])
            dz = np.dot(w.T,dz_next) * self.actuator(z_str,derive = 1)
            dz = dz.reshape(z.shape)
        elif type(next_layer) == max_pool_layer:
            ind_mat = conv_indices(z,[next_layer.lp[1][0],next_layer.lp[1][1],next_layer.lp[1][3]]) 
            x_max_ind = next_layer.x_max_ind
            dz = dz_pool(z,x_max_ind,ind_mat,dz_next)
        elif type(next_layer) == conv_layer:
            ind_mat = conv_indices(z,[w.shape[0],w.shape[1],self.stride]) 
            dz= dz_calc(z,w,ind_mat,dz_next)
            dz = dz* self.actuator(z,derive = 1)
        m = z.shape[-1]
        fh = self.lp[1][0]
        fw = self.lp[1][1]
        dw = dw_calc(a0,dz,[fh,fw,self.stride])/m
        db = np.sum(dz,axis = -1,keepdims = True)/m
        return dz,dw,db  

 
class fully_connected_layer():
    def __init__(self,actuator,layer_sizes):
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        self.actuator = actuator
        #if len (layer_sizes[0]) >1:
        if type(layer_sizes[0]) == list:
            layer_sizes[0] = np.prod(layer_sizes[0])
        self.ls = layer_sizes
    def initialize(self):
        ls = self.ls
        signs = (2*np.random.randint(0,2,size=self.ls[1]*ls[0] )-1).reshape(ls[1],ls[0] )
        var = np.sqrt(2/ls[1])
        w0 = var* 1*signs*((np.random.randint( 10,1e2, size=ls[1]*ls[0] )/1e2 ) ).reshape( [ls[1],ls[0]] )
        b0 = np.zeros([ls[1],1])
        return w0,b0
    def fp(self,w,b,a0):
        if len(a0.shape) > 2:
            a0 = a0.reshape(self.ls[0],-1)
        z = np.dot(w,a0) + b 
        a = self.actuator(z,derive = 0)
        return z,a
    def bp(self,w,dz_next,z,a0,next_layer = None):
        #dz [L] : w[L+1],dz[L+1],z[L],a[L-1]
        if len(a0.shape) > 2:
            a0 = a0.reshape(self.ls[0],-1)
        m = z.shape[1]
        dz = np.dot(w.T,dz_next) * self.actuator(z,derive = 1)
        db = np.sum(dz, axis = 1).reshape(dz.shape[0],1)/m
        dw = np.dot(dz,a0.T)/m
        return dz,dw,db  

# %% Network class
        
class network():
    def __init__(self,epochs,tolerance,actuators,layer_parameters,layer_types,alpha,beta1,beta2,epsilon,gamma,lam):
        self.epochs = epochs
        self.tol = tolerance
        self.actuators = actuators
        self.layer_types = layer_types
        self.layer_parameters = layer_parameters
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        
    def initialize(self):
        wv = [np.array([[9999],[9999]])]
        bv = [np.array([[9999],[9999]])]
        self.Vw = [np.array([[0],[0]])]
        self.Vb = [np.array([[0],[0]])]
        self.Sw = [np.array([[0],[0]])]
        self.Sb = [np.array([[0],[0]])]
        ls = self.ls
        for L in range(1,len(ls)):
            w0,b0 = self.layers[L].initialize()
            wv.append(w0)
            bv.append(b0)
            self.Vw.append(np.zeros_like(w0))
            self.Vb.append(np.zeros_like(b0))
            self.Sw.append(np.zeros_like(w0))
            self.Sb.append(np.zeros_like(b0))
        return wv,bv

    def Forward_prop(self,x,wv,bv):
        a = [x] # Zero'th member of a is the input
        z= [[0]] # z[0] isn't actually used, it's just added to sync with a
        for L in range(1,len(wv)):
            z0,a0 = self.layers[L].fp(wv[L],bv[L],a[L-1])
            z.append(z0)
            a.append(a0)
        return a,z

    def Back_prop(self,y,a,z,wv):
        ls = self.ls
        m = a[-1].shape[-1]
        ## NOTE: dz(outputs,samples) { same dimensions as Z}
        dz = []
        dz_last = a[-1]-y # NOTE: This is outside of loop because of the cost/softmax functions
        dz.insert(0,dz_last)
        db = [np.sum(dz[0], axis = 1).reshape(ls[-1],1)/m] # Reshape is required for broadcasting #CAREFUL
        dw = [np.dot(dz[0],a[-2].T)/m]
        for L in range(2,len(wv)): # NOTE: this loop doesn't run over the last layer, since its already been calculated
            # NOTE: dz[0] starts as dz[ind+1] in this loop
            ind = len(wv)-L # NOTE: counts from the end of the list, starts at len(ls)-1
            dz_temp,dw_temp,db_temp = self.layers[ind].bp(wv[ind+1],dz[0],z[ind],a[ind-1],self.layers[ind+1])
            dz.insert(0,dz_temp)
            db.insert(0,db_temp)
            dw.insert(0,dw_temp)
        dz.insert(0,[0])
        db.insert(0,[0])
        dw.insert(0,[0])
        return dz,dw,db


    def CurrentDecay(t):
        I0 = self.epochs*self.batch_size
        A0 = self.gamma
        decay = np.exp(1/I0*np.log(A0)*t)
    def Optimization_step(self,wv,bv,dw,db,decay,t):
        m = self.m
        self.alpha_tot = self.alpha * decay
        for L in range(1,len(wv)):
            self.Vw[L] = self.beta1*self.Vw[L]+(1-self.beta1)*dw[L]
            self.Vb[L] = self.beta1*self.Vb[L]+(1-self.beta1)*db[L]
            self.Sw[L] = self.beta2*self.Sw[L]+(1-self.beta2)*dw[L]**2
            self.Sb[L] = self.beta2*self.Sb[L]+(1-self.beta2)*db[L]**2
            Vw_corrected = self.Vw[L] / (1-self.beta1**t)
            Vb_corrected = self.Vb[L] / (1-self.beta1**t)
            Sw_corrected = self.Sw[L] / (1-self.beta2**t)
            Sb_corrected = self.Sb[L] / (1-self.beta2**t)
            #wv[L] =(1-self.lam*self.alpha_tot/m)*wv[L]-self.alpha_tot * Vw_corrected/(Sw_corrected**(1/2)+self.epsilon)
            wv[L] =(1-self.lam*self.alpha_tot/m)*wv[L]-self.alpha_tot * Vw_corrected/(Sw_corrected**(1/2)+self.epsilon)
            bv[L]+=-self.alpha_tot * Vb_corrected/(Sb_corrected**(1/2)+self.epsilon)
        return wv,bv
    
    def Total_cost(self,y,a):
        eps = 1e-8
        cost = -(y*np.log(a[-1]+eps))
        m = y.shape[1]
        Ltot = 1/m*sum(sum(cost))
        return Ltot
        
    def Grad_check(self,wv,bv,dw,db,x_batch, y_batch):
        eps = 1e-9
        i = 2
        j = 1
        channel = None
        filt = None
        L = -1
        wv1 = copy.deepcopy(wv)
        wv1[L][i,j,channel,filt]-=eps
        a1,_ = self.Forward_prop(x_batch,wv1,bv)
        wv2 = copy.deepcopy(wv)
        wv2[L][i,j,channel,filt]+=eps
        a2,_ = self.Forward_prop(x_batch,wv2,bv)
        cost1 = self.Total_cost(y_batch,a1)
        cost2 = self.Total_cost(y_batch,a2)
        dw_approx = (cost2-cost1)/(2*eps)
        error =(dw_approx-dw[L][i,j,channel,filt])/(np.abs(dw_approx)+np.abs(dw[L][i,j,channel,filt]))
        print('grad check error',error)
        return dw_approx    
    
    def normalize(self,x):
        # x = x[inputs,samples]
        x_mean = np.mean(x,axis=-1,keepdims = True)
        x_new = x - x_mean
        x_std  = RMS(x_new, ax = -1,kdims = True)
        x_new /= (x_std+self.epsilon)
        return x_new,x_mean,x_std
    
    def setupLayerSizes(self,x,y):
        self.m = x.shape[-1] #number of samples
        # Create network layers
        if len(x.shape)>2:
            inputs = [[x.shape[0], x.shape[1], x.shape[2]]]
        else:
            inputs = [[x.shape[0]]]
        outputs = [[y.shape[0]]]
        self.lp = inputs+ self.layer_parameters+ outputs #layer parameters
        
        layer_sizes = inputs[0]
        if len(x.shape)>2:
            zw =[ x.shape[0] ]
            zh =[ x.shape[1]]
            filts = [x.shape[2]]
            layer_sizes = [[zw[0],zh[0],filts[0]]]
        for l,lpar in enumerate(self.lp[1:],1):
            if len(lpar)>1:
                # then layer is not FCL
                # lpar :[ fh,fw , filters, stride]
                stride = lpar[3]
                zh_temp  = int(np.floor((zh[l-1] - lpar[1])/stride+1))
                zw_temp  = int(np.floor((zw[l-1] - lpar[0])/stride+1))
                zh.append( zh_temp)
                zw.append( zw_temp)
                filts.append(lpar[2])
                layer_sizes.append([zw[l],zh[l],filts[l]])
            else:
                layer_sizes.append(lpar[0])
        self.ls = layer_sizes
        
        self.layer_types= [0] + self.layer_types + ['fc']
        self.layers = [0]
        for n,act in enumerate(self.actuators[1:],1):
            layer_sizes = [self.ls[n-1],self.ls[n]]
            layer_parameters = [self.lp[n-1],self.lp[n]]
            if self.layer_types[n] == 'fc':
                self.layers.append(fully_connected_layer(act,layer_sizes))
            elif self.layer_types[n] == 'conv':
                self.layers.append(conv_layer(act,layer_sizes,layer_parameters))   
            elif self.layer_types[n] == 'max_pool':
                self.layers.append(max_pool_layer(layer_sizes,layer_parameters))   

    def train(self,x,y,batch_size = None,wv = None, bv = None):
        # Setup layer sizes, batches, normalize inputs
        if batch_size == None or batch_size >= x.shape[-1]:
            self.batch_size = x.shape[-1]
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size
        self.x,self.x_mean,self.x_std = self.normalize(x)
        self.y = y
        
        self.setupLayerSizes(x,y)        
        # Initialize network parameters
        wv0,bv0,Vw,Vb,Sw,Sb = self.initialize() 
        if wv == None:
            wv = wv0
        if bv == None:
            bv = bv0
            
        # Begin optimization iterations
        last_cost_mean = self.tol*2
        Ltot = 0
        epoch = 0
        self.J = []
        batches = int(np.floor(self.x.shape[-1]/batch_size))
        while last_cost_mean>self.tol and epoch <self.epochs: 
            for t in range(batches):
                if len(self.x.shape)>2:
                    x_batch = self.x[:,:,:,t*batch_size:(t+1)*batch_size]
                else:
                    x_batch = self.x[:,t*batch_size:(t+1)*batch_size]
                y_batch = self.y[:,t*batch_size:(t+1)*batch_size]
                
                a,z = self.Forward_prop(x_batch,wv,bv)
                
                dz,dw,db = self.Back_prop(y_batch,a,z,wv)
                
                t_tot = ((epoch+1)*t+1) # t paramater for average correction

                # Periodic operations
                if not epoch % 1:
#                    dw_approx = self.Grad_check(wv,bv,dw,db,x_batch,y_batch)
                    print('Epoch#: {0:d} batch#: {1:d}'.format(epoch,t))
                    print('batch.cost: {0:.4f} , mean.cost:{1:.4f} Ptot: {2:.4f}'.format(Ltot,last_cost_mean,np.exp(-Ltot)))
                
                wv,bv = self.Optimization_step(wv,bv,dw,db,t_tot)
                Ltot = self.Total_cost(y_batch,a)
                self.J.append(Ltot)
                #time.sleep(1e-6) # Enables to break midloop, only for programming stage
                
            
            last_cost_mean = np.mean(self.J[-batches:])
            print('Cost mean:',last_cost_mean)
            epoch+=1
        # end iterations
        self.w = wv
        self.b = bv
        self.z = z
        self.a = a
        self.dz = dz
        self.dw = dw
        self.db = db
        
    def predict(self,x):
        x_test = x-self.x_mean
        x_test /= (self.x_std+self.epsilon)
        a,_ = self.Forward_prop(x_test,self.w,self.b)
        y = np.zeros_like(a[-1])
        y[a[-1].argmax(0),np.arange(a[-1].shape[1])] = 1 # Best guess for y, based on max probability
        return a,y


