# %% Network class
from LayerClasses import *
import time
import copy #for gradcheck only
import pdb

def RMS(x,ax = None,kdims = False):
    y = np.sqrt(np.mean(x**2,axis = ax,keepdims = kdims ))
    return y

class network():
    def __init__(self,epochs,tolerance,
                 actuators,layer_parameters,layer_types,
                 learningRate,beta1,beta2,epsilon,lam,learningDecay,costFunctionType='xEntropy',dzFunc=None):

        self.epochs = epochs
        self.tol = tolerance
        
        self.actuators = actuators
        self.layer_types = layer_types
        self.layer_parameters = layer_parameters

        self.learningRate = learningRate # Step size
        self.beta1 = beta1 # Step weighted average parameter
        self.beta2 = beta2 # Step normalization parameter
        self.epsilon = epsilon # Addition to denominator to prevent div by 0
        self.lam = lam # Regularization parameter

        self.learningDecay = learningDecay

        self.costFunctionType = costFunctionType
        
        self.wv = None
        self.bv = None
        self.dz = None
        
    def InitializeWeights(self):
        # Initialize network parameters 
        wv = [np.array([[0.0],[0.0]])]
        bv = [np.array([[0.0],[0.0]])]
        self.Vw = [np.array([[0.0],[0.0]])]
        self.Vb = [np.array([[0.0],[0.0]])]
        self.Sw = [np.array([[0.0],[0.0]])]
        self.Sb = [np.array([[0.0],[0.0]])]
        ls = self.ls
        for L in range(1,len(ls)):
            w0,b0 = self.layers[L].initialize()
            wv.append(w0)
            bv.append(b0)
            self.Vw.append(np.zeros_like(w0))
            self.Vb.append(np.zeros_like(b0))
            self.Sw.append(np.zeros_like(w0))
            self.Sb.append(np.zeros_like(b0))
        # if pickled parameters weren't provided, use random intial guess
        if self.wv == None:
            self.wv = wv
        if self.bv == None:
            self.bv = bv

    def ForwardProp(self,x):
        a = [x] # Zero'th member of a is the input
        z= [[0]] # z[0] isn't actually used, it's just added to sync dimensions with 'a'
        for L in range(1,len(self.wv)):
            z0,a0 = self.layers[L].fp(self.wv[L],self.bv[L],a[L-1])
            z.append(z0)
            a.append(a0)
        return a,z

    def BackProp(self,y,a,z,dzFunc = 'Softmax/xEntropy'):
        ls = self.ls
        m = a[-1].shape[-1]
        ## NOTE: dz(outputs,samples) { same dimensions as Z}
        dz = []
        # dzFunc is dL/dz = dL/da*da/dz=self.actuators[-1](z[-1],1)
        if dzFunc == 'Softmax/xEntropy':
            dz_last = a[-1]-y # NOTE: This is outside of loop because of the cost/softmax functions
        elif dzFunc == 'Linear/L2':
            dz_last = 2*(a[-1]-y)
        else:
            dz_last = dzFunc(a[-1],z[-1],y,self.actuators[-1])
            
        dz.insert(0,dz_last)
        db = [np.sum(dz[0], axis = 1).reshape(ls[-1],1)/m] # Reshape is required for broadcasting #CAREFUL
        
        dw = [np.dot(dz[0],a[-2].T)/m]
        for L in range(2,len(self.wv)): # NOTE: this loop doesn't run over the last layer, since its already been calculated above
            # NOTE: dz[0] starts as dz[ind+1] in this loop
            ind = len(self.wv)-L # NOTE: counts from the end of the list, starts at len(ls)-1
            dz_temp,dw_temp,db_temp = self.layers[ind].bp(self.wv[ind+1],dz[0],z[ind],a[ind-1],self.layers[ind+1])
            dz.insert(0,dz_temp)
            db.insert(0,db_temp)
            dw.insert(0,dw_temp)
        dz.insert(0,[0])
        db.insert(0,[0])
        dw.insert(0,[0])
        self.dz = dz
##        pdb.set_trace()
        return dz,dw,db
    
    def OptimizationStepADAM(self,dw,db,t):
        # ADAM optimization algorithm
        self.alpha_tot = self.learningRate * self.learningDecay**t # Decay learning rate
        for L in range(1,len(self.wv)):
            self.Vw[L] = self.beta1*self.Vw[L]+(1-self.beta1)*dw[L]
            self.Vb[L] = self.beta1*self.Vb[L]+(1-self.beta1)*db[L]
            self.Sw[L] = self.beta2*self.Sw[L]+(1-self.beta2)*dw[L]**2
            self.Sb[L] = self.beta2*self.Sb[L]+(1-self.beta2)*db[L]**2
            Vw_corrected = self.Vw[L] / (1-self.beta1**t)
            Vb_corrected = self.Vb[L] / (1-self.beta1**t)
            Sw_corrected = self.Sw[L] / (1-self.beta2**t)
            Sb_corrected = self.Sb[L] / (1-self.beta2**t)
            #wv[L] =(1-lam*self.alpha_tot/self.m)*wv[L] - self.alpha_tot * Vw_corrected/(Sw_corrected**(1/2)+self.epsilon)
            self.wv[L] = (1-self.lam)*self.wv[L] - self.alpha_tot * Vw_corrected/(Sw_corrected**(1/2)+self.epsilon)
            self.bv[L] = self.bv[L] - self.alpha_tot * Vb_corrected/(Sb_corrected**(1/2)+self.epsilon)
    def OptimizationStep(self,dw,db,t):
        # Direct step
        for L in range(1,len(self.wv)):
            self.wv[L] += -self.learningRate*dw[L]
            self.bv[L] += -self.learningRate*db[L]
    def TotalCost(self,y,a):
        eps = 1e-8
        # Sum up cross entropy costs
        if self.costFunctionType == 'xEntropy':
            cost = -(y*np.log(a[-1]+eps))
        elif self.costFunctionType == 'L2':
            cost = (a[-1]-y)**2
        m = y.shape[1]
        Ltot = 1/m*sum(sum(cost))
        return Ltot
        
    def GradCheck(self,dw,db,x_batch, y_batch):
        '''
         This function runs a simple numerical differentiation of loss regarading the weights w
         to compare with the analytical gradient calculation
         Used for debugging
        '''
        eps = 1e-5
        i = 2
        j = 1
        channel = None
        filt = None
        L = -2
        self.wv[L][i,j,channel,filt]-=eps
        a1,_ = self.ForwardProp(x_batch)
        self.wv[L][i,j,channel,filt]+=eps
        a2,_ = self.ForwardProp(x_batch)
        cost1 = self.TotalCost(y_batch,a1)
        cost2 = self.TotalCost(y_batch,a2)
        dw_approx = (cost2-cost1)/(eps)
##        error =(dw_approx-dw[L][i,j,channel,filt])/(np.abs(dw_approx)+np.abs(dw[L][i,j,channel,filt]))
        dw_net= dw[L][i,j]
        error =(dw_approx-dw_net)/(np.abs(dw_approx)+np.abs(dw_net))
        print('grad check error {:1.3f}%'.format(error*100))
        
        return dw_approx    
    
    def Normalize(self,x):
        # x = x[inputs,samples]
        x_mean = np.mean(x,axis=-1,keepdims = True)
        x_new = x - x_mean
        x_std  = RMS(x_new, ax = -1,kdims = True)
        x_new /= (x_std+self.epsilon)
        return x_new,x_mean,x_std     
                
    def SetupLayerSizes(self,x,y):
        self.m = x.shape[-1] #number of samples
        # Create network layers
        if len(x.shape)>2:
            inputs = [[x.shape[0], x.shape[1], x.shape[2]]]
        else:
            inputs = [[x.shape[0]]]
        outputs = [[y.shape[0]]]

        # Layer parameters list (Dimensions of filters or weight matrices)
        self.lp = inputs+ self.layer_parameters+ outputs #layer parameters
        
        # Layer dimensions
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
        self.SetupLayerSizes(x,y)

        # Load provided weights
        if wv != None:
            self.wv = wv
        if bv != None:
            self.bv = bv
        
        # Initialize network parameters
        self.InitializeWeights()
            
        # Begin optimization iterations
        last_cost_mean = self.tol*2 # initial value, to start the loop
        Ltot = 0
        epoch = 0
        self.J = [] # Loss list
        batches = int(np.floor(self.x.shape[-1]/batch_size)) #Number of batches per epoch
        while last_cost_mean>self.tol and epoch <self.epochs: 
            for t in range(batches):
                # Organize batch input/output
                if len(self.x.shape)<=2:
                    x_batch = self.x[:,t*batch_size:(t+1)*batch_size]
                else:
                    x_batch = self.x[:,:,:,t*batch_size:(t+1)*batch_size]
                y_batch = self.y[:,t*batch_size:(t+1)*batch_size]


                a,z = self.ForwardProp(x_batch)
                
                dz,dw,db = self.BackProp(y_batch,a,z)
                for L in range(1,len(self.wv)):
                    if np.isnan(dw[L]).any():
                        import pdb
                        pdb.set_trace()
                
                t_tot = ((epoch+1)*t+1) # t paramater for average correction

                # Periodic operations
##                if t % 10 == 0:
##                    print('Epoch#: {0:d} batch#: {1:d}'.format(epoch,t))
##                    print('batch.cost: {0:.4f} , mean.cost:{1:.4f} Ptot: {2:.4f}'.format(Ltot,last_cost_mean,np.exp(-Ltot)))
                
                self.OptimizationStep(dw,db,t=t_tot)
                Ltot = self.TotalCost(y_batch,a)
                self.J.append(Ltot)
                #time.sleep(1e-6) # Enables to break midloop, only for debugging
                
            # end batches
            last_cost_mean = np.mean(self.J[-batches:])
            wSize= 0
            for L in range(1,len(self.wv)):
                wSize += np.mean(self.wv[L]**2)
            wSize = np.sqrt(wSize)
            pred = self.Predict(x)
            # Prediction guess at maximal probability
            yOneHot = np.zeros_like(pred)
            yOneHot[pred[-1].argmax(0),np.arange(pred[-1].shape[1])] = 1
            accuracy = np.mean(np.min((y==pred),axis=0))*100
            print('Epoch {0:3.0f}  ;  wSize {1:1.5f}  ;   Cost {2:1.5f}'.format(epoch,wSize,last_cost_mean))
            print('a',accuracy)
            epoch+=1
            
        # end iterations
        # Save parameters and network calculations 
        self.z = z
        self.a = a
        self.dz = dz
        self.dw = dw
        self.db = db
        
    def Predict(self,x):
        # Normallize input
        if hasattr(self, 'x_mean'):
            x_test = x-self.x_mean
            x_test /= (self.x_std+self.epsilon)
        else:
            x_test = x
        a,_= self.ForwardProp(x_test)
        pred = a[-1]
        return pred



def ExampleNet():
    ## Define Neural Network policy approximator
    ## Hyper parameters


    epochs = 20 # Irrelevant to RL
    tolerance = 1e-5 # Irrelevant to RL

    layer_parameters = [[200],[50] ]
    layer_types = ['fc','fc']
    actuators = [[0] ,ReLU2,ReLU2,Softmax]
    learningRate = 0.005 # Learning Rate, this is just a temporary placeholder, the actual value is defined in the main loop
    beta1 = 0.95 # Step weighted average parameter
    beta2 = 0.999 # Step normalization parameter
    epsilon = 1e-10 # Addition to denominator
    lam = 1e-5 # Regularization parameter
    learningDecay = 0.95
    neuralNetwork = network(epochs,tolerance,
                     actuators,layer_parameters,layer_types,
                     learningRate,beta1,beta2,epsilon,lam,learningDecay,dzFunc=None)
    return neuralNetwork
