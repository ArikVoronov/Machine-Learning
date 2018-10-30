# %% Network class
from LayerClasses import *
import time

def RMS(x,ax = None,kdims = False):
    y = np.sqrt(np.mean(x**2,axis = ax,keepdims = kdims ))
    return y

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
        # Initialize network parameters 
        wv = [np.array([[9999],[9999]])]
        bv = [np.array([[9999],[9999]])]
        Vw = [np.array([[0],[0]])]
        Vb = [np.array([[0],[0]])]
        Sw = [np.array([[0],[0]])]
        Sb = [np.array([[0],[0]])]
        ls = self.ls
        for L in range(1,len(ls)):
            w0,b0 = self.layers[L].initialize()
            wv.append(w0)
            bv.append(b0)
            Vw.append(np.zeros_like(w0))
            Vb.append(np.zeros_like(b0))
            Sw.append(np.zeros_like(w0))
            Sb.append(np.zeros_like(b0))
        return wv,bv,Vw,Vb,Sw,Sb

    def Forward_prop(self,x,wv,bv):
        a = [x] # Zero'th member of a is the input
        z= [[0]] # z[0] isn't actually used, it's just added to sync dimensions with 'a'
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
        for L in range(2,len(wv)): # NOTE: this loop doesn't run over the last layer, since its already been calculated above
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
    
    def Optimization_step(self,wv,bv,dw,db,Vw,Vb,Sw,Sb,t):
    def Optimization_step(self,wv,bv,dw,db,t):
        # ADAM optimization algorithm
        m = self.m
        # Decay step size
        I0 = self.epochs*self.batch_size
        A0 = self.gamma
        self.alpha_tot = self.alpha * np.exp(1/I0*np.log(A0)*t)
        
        for L in range(1,len(wv)):
            self.Vw[L] = self.beta1*self.Vw[L]+(1-self.beta1)*dw[L]
            self.Vb[L] = self.beta1*self.Vb[L]+(1-self.beta1)*db[L]
            self.Sw[L] = self.beta2*self.Sw[L]+(1-self.beta2)*dw[L]**2
            self.Sb[L] = self.beta2*self.Sb[L]+(1-self.beta2)*db[L]**2
            Vw_corrected = self.Vw[L] / (1-self.beta1**t)
            Vb_corrected = self.Vb[L] / (1-self.beta1**t)
            Sw_corrected = self.Sw[L] / (1-self.beta2**t)
            Sb_corrected = self.Sb[L] / (1-self.beta2**t)
            #wv[L] =(1-lam*self.alpha_tot/m)*wv[L]-self.alpha_tot * Vw_corrected/(Sw_corrected**(1/2)+self.epsilon)
            wv[L] =(1-self.lam)*wv[L]-self.alpha_tot * Vw_corrected/(Sw_corrected**(1/2)+self.epsilon)
            bv[L]+=-self.alpha_tot * Vb_corrected/(Sb_corrected**(1/2)+self.epsilon)
        return wv,bv
    
    def Total_cost(self,y,a):
        # Sum up cross entropy costs
        eps = 1e-8
        cost = -(y*np.log(a[-1]+eps))
        m = y.shape[1]
        Ltot = 1/m*sum(sum(cost))
        return Ltot
        
    def Grad_check(self,wv,bv,dw,db,x_batch, y_batch):
        '''
         This function runs a simple numerical differentiation of loss regarading the weights w
         to compare with the analytical gradient calculation
         Used for debugging
        '''
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
    
    def Normalize(self,x):
        # x = x[inputs,samples]
        x_mean = np.mean(x,axis=-1,keepdims = True)
        x_new = x - x_mean
        x_std  = RMS(x_new, ax = -1,kdims = True)
        x_new /= (x_std+self.epsilon)
        return x_new,x_mean,x_std
    
    def OrganizeLayers(self):
        if len(self.x.shape)<=1:
            # If inputs are vectors
            inputs = [[self.x.shape[0]]]
        else:
           # If inputs are images
            inputs = [[self.x.shape[0], self.x.shape[1], self.x.shape[2]]]
            
        outputs = [[self.y.shape[0]]]

        # Layer parameters list (Dimensions of filters or weight matrices)
        self.lp = inputs+ self.layer_parameters+ outputs

        # Layer dimensions
        layer_sizes = inputs[0]
        if len(self.x.shape)>2:
            zw =[ self.x.shape[0] ]
            zh =[ self.x.shape[1]]
            filts = [self.x.shape[2]]
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

        # List of layers
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
        
    def Train(self,x,y,batch_size = None,wv = None, bv = None):
        if batch_size == None or batch_size >= x.shape[-1]:
            self.batch_size = x.shape[-1]
            batch_size = self.batch_size
        else:
            self.batch_size = batch_size
        self.x,self.x_mean,self.x_std = self.Normalize(x)
        self.y = y
        self.m = x.shape[-1] #number of samples
        
        # Create network layers
        self.OrganizeLayers()

                
        # Initialize network parameters
        wv0,bv0,Vw,Vb,Sw,Sb = self.initialize()
        # if pickled parameters weren't provided, use random intial guess
        if wv == None:
            wv = wv0
        if bv == None:
            bv = bv0
            
        # Begin optimization iterations
        last_cost_mean = self.tol*2 # initial value, to start the loop
        Ltot = 0
        epoch = 0
        self.J = [] # Loss list
        batches = int(np.floor(self.x.shape[-1]/batch_size)) #Number of batches per epoch
        while last_cost_mean>self.tol and epoch <self.epochs: 
            for t in range(batches):
                # Organize batch input/output
                if len(self.x.shape)<=1:
                    x_batch = self.x[:,t*batch_size:(t+1)*batch_size]
                else:
                    x_batch = self.x[:,:,:,t*batch_size:(t+1)*batch_size]
                y_batch = self.y[:,t*batch_size:(t+1)*batch_size]

                a,z = self.Forward_prop(x_batch,wv,bv)
                
                dz,dw,db = self.Back_prop(y_batch,a,z,wv)
                
                t_tot = ((epoch+1)*t+1) # t paramater for average correction

                # Periodic operations
                if not epoch % 1:
                    print('Epoch#: {0:d} batch#: {1:d}'.format(epoch,t))
                    print('batch.cost: {0:.4f} , mean.cost:{1:.4f} Ptot: {2:.4f}'.format(Ltot,last_cost_mean,np.exp(-Ltot)))
                
                wv,bv,Vw,Vb,Sw,Sb = self.Optimization_step(wv,bv,dw,db,Vw,Vb,Sw,Sb,t_tot)
                Ltot = self.Total_cost(y_batch,a)
                self.J.append(Ltot)
                #time.sleep(1e-6) # Enables to break midloop, only for debugging
            # end batches
            last_cost_mean = np.mean(self.J[-batches:])
            print('Cost mean:',last_cost_mean)
            epoch+=1
        # end iterations
        # Save parameters and network calculations 
        self.w = wv
        self.b = bv
        self.z = z
        self.a = a
        self.dz = dz
        self.dw = dw
        self.db = db
        
    def predict(self,x):
        # Normallize input
        x_test = x-self.x_mean
        x_test /= (self.x_std+self.epsilon)
        a,_ = self.Forward_prop(x_test,self.w,self.b)
        # Prediction guess at maximal probability
        y = np.zeros_like(a[-1])
        y[a[-1].argmax(0),np.arange(a[-1].shape[1])] = 1
        return a,y
