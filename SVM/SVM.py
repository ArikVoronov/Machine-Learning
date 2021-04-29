import numpy as np
import cvxopt 
from QPOpt import *

class SVM():
    def __init__(self,C=None,dualForm = True,useCVX = True):
        self.C = C # larger values decrease the allowable margin, NONE = Hard Margin
        self.dualForm = dualForm
        self.useCVX = useCVX
    def SetupDualMatrices(self,X,y):
        '''
        Formulate matrices and vectors for a dual QP problem.
        min 1/2 x.T*P*x + q.T*x 
        s.t. Gx <= h 
             Ax = b 
        
        Where variables x are the Lagrangian Multipliers of the original problem.

        Parameters
        ----------
        X : np.array(samples,features); input data.
        y : np.array(samples,features); data labels.

        Returns
        -------
        P,q,A,b,G,h : dual problem QP matrices and vectores

        '''
        # 1/2aHa-sum(a)
        X_dash = y * X
        P = np.dot(X_dash , X_dash.T) * 1.
        q = -np.ones((self.n_samples, 1))
        if self.C is None: # hard margin
            #-a<=0
            G = -np.eye(self.n_samples)
            h = np.zeros((self.n_samples,1))
        else:
            #-a<=0 ;a<=C
            G = np.vstack((-np.eye(self.n_samples), np.eye(self.n_samples)))
            h = np.hstack((np.zeros(self.n_samples), np.ones(self.n_samples) * self.C))
        # sum(y*a)=0
        A = y.reshape(1, self.n_samples)
        b = np.zeros((1,1))
        return P,q,A,b,G,h 
    def SetupMatrices(self,X,y):
        '''
        Formulate matrices and vectors for a QP regular representation.
        min 1/2 x.T*P*x + q.T*x 
        s.t. Gx <= h 
             Ax = b 
        
        Where x are concatinated w and b (and xi for soft margin) SVM parameters.
        

        Parameters
        ----------
        X : np.array(samples,features); input data.
        y : np.array(samples,features); data labels.

        Returns
        -------
        P,q,A,b,G,h : regular QP matrices and vectores

        '''
        Xb = np.hstack([X,np.ones([self.n_samples,1])])
        GM = -y*Xb
        if self.C == None: # hard margin
            q = np.zeros((self.n_features+1, 1))
            P = np.eye(self.n_features+1)
            P[-1][-1]=0
            G = GM
            h = -np.ones(self.n_samples)[:,None]
        else:
            P = np.zeros([self.n_features+1+self.n_samples,self.n_features+1+self.n_samples])
            for i in range(self.n_features):
                P[i][i]=1.0
            q = np.vstack((np.zeros((self.n_features+1, 1)),self.C*np.ones([self.n_samples,1])))
            G = np.hstack([GM,-np.eye(self.n_samples)])
            xiMat = np.hstack([np.zeros_like(GM),-np.eye(self.n_samples)])
            G = np.vstack([G,xiMat])
            h = np.vstack([-np.ones([self.n_samples,1]),np.zeros([self.n_samples,1])])
        A=np.array([])
        b=np.array([])
        return P,q,A,b,G,h 
    def fit(self,X,y):
        '''
        Solve the QP optimization for dual/regular forms.
        The result SVM parameters w and b are stored in the object,
        predict, classify and accuracy methods can be runs afterwards.

        Parameters
        ----------
        X : np.array(samples,features); input data.
        y : np.array(samples,features); data labels.

        '''
        self.n_samples,self.n_features = X.shape
        if self.dualForm:
            P,q,A,b,G,h = self.SetupDualMatrices(X,y)
        else:
            P,q,A,b,G,h = self.SetupMatrices(X,y)
        if self.useCVX:
            self.solX = self.CVXOptimization(P,q,A,b,G,h)
        else:
            self.solX = QuadOptimization(P,q,A,b,G,h)
        if self.dualForm:
            self.alphas = np.array(self.solX)
            self.w = np.squeeze(np.dot((self.alphas*y).T,X))
            outliers = self.GetOutliers(self.alphas)# get boundary points
            outliers = np.round(self.alphas,8)>0
            if C is None:
                xs = X[outliers,:]; ys = y[outliers,:]
                self.b = ys[0]-np.dot(xs[0],self.w.T) # for nonzero x : b = 1/y_i-wx
            else:
                outs=(outliers*(self.alphas<self.C)).squeeze()
                # outs = [o for o in outliers if self.alphas[o]<self.C]
                xs = X[outs,:]; ys = y[outs,:]
                self.b = ys[0]-np.dot(xs[0],self.w.T) # for nonzero x : b = 1/y_i-wx
        else:
            self.w = self.solX[:self.n_features]
            self.b = self.solX[self.n_features]
        
    def CVXOptimization(self,P,q,A,b,G,h):
        '''
        CVXopt wrapper for quadratic programming optimization.
        Works for either dual or regular forms.

        Parameters
        ----------
        Input matrices and vectors as required by a convex optimization problem
        min 1/2 x.T*P*x + q.T*x 
        s.t. Gx <= h 
             Ax = b 

        Returns
        -------
        solX : np.array(samples,features); Solution of the optimization problem.
        '''
        Pc = cvxopt.matrix(P)
        qc = cvxopt.matrix(q)
        if A.size>0:
            Ac = cvxopt.matrix(A)
            bc = cvxopt.matrix(b)
        else:
            Ac = None
            bc = None
        Gc = cvxopt.matrix(G)
        hc = cvxopt.matrix(h)
        #Setting solver parameters (change default to decrease tolerance) 
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = 1e-20
        cvxopt.solvers.options['reltol'] = 1e-20
        cvxopt.solvers.options['feastol'] = 1e-20
        #Run solver
        sol = cvxopt.solvers.qp(Pc, qc,Gc, hc,Ac,bc)
        solX = sol['x']
        return solX
   
    def predict(self,x):
        yPredict = np.dot(x,self.w)+self.b
        return yPredict[:,None]
        
    def classify(self,x):
        '''
        Parameters
        ----------
        x : np.array(samples,features); datapoints to classify
        
        Returns
        -------
        yClass : np.array(samples,1); respective labels for input data
        '''
        yPredict = self.predict(x)
        yClass = np.zeros_like(yPredict)
        yClass[yPredict>=0] = 1
        yClass[yPredict<0] = -1
        yClass = yClass
        return yClass

    def accuracy(self,x,y):
        '''
        Parameters
        ----------
        x : np.array(samples,features); datapoints to classify
        y : np.array(samples,1); actual labels

        Returns
        -------
        accuracy : scalar (float); rate of accuarte classification
        '''
        samples = len(y)
        yClass = self.classify(x)
        accuracy = np.sum(y.squeeze()==yClass.squeeze())/samples
        return accuracy
    
    def GetOutliers(self,a):
        '''
        This is used to find outliers in alphas for the dual problem.
        DEPRECATED; now just use all alphas>0.

        Parameters
        ----------
        a : np.array; array of numeric variables

        Returns
        -------
        outliers : np.array; array of statistical outliers
        '''
        outliersCount = 0
        tol = 1e4
        while outliersCount == 0:
            q75, q25 = np.percentile(a, [75 ,25])
            iqr = q75 - q25
            outliers = (a>q75*3*tol).squeeze()
            outliersCount = sum(outliers) # sum of boolians gives a count
            tol/=10
        return outliers

def Create2DSet(samples):
    '''
    Parameters
    ----------
    samples : int; number of samples (total).

    Returns
    -------
    shuffled_X : np.array(samples,features); input data.
    shuffled_y : np.array(samples,features); data labels.
    '''
    assert(samples%2==0)
    sigx = 0.6; sigy = 0.6
    x_pos = np.random.multivariate_normal([3,4],[[sigx,0],[0,sigy]],int(samples/2))
    x_neg = np.random.multivariate_normal([2,2],[[sigx,0],[0,sigy]],int(samples/2))
    X = np.vstack((x_neg, x_pos))
    y = np.concatenate((-1*np.ones([len(x_pos),1]),1*np.ones([len(x_neg),1])))
    y = y.reshape(-1,1) * 1.

    #shuffle
    p = np.random.permutation(y.shape[0])
    shuffled_X = X[p,:]
    shuffled_y = y[p]
    return shuffled_X,shuffled_y

def Plot2DSet(X,y,w=None,b=None,drawHyperplane=False):
    ''' Plot wrapper for 2D SVM, show labled dataset and SVM hyperplane '''
    fig = plt.figure(figsize = (10,10))
    for x1,y1 in zip(X,y):
        if y1 ==-1:
            plt.scatter(x1[0], x1[1], marker = 'x', color = 'r')
        else:
            plt.scatter(x1[0], x1[1], marker = 'o', color = 'b')
    xmin = np.floor(np.min(X))*0; xmax = np.ceil(np.max(X))
    if drawHyperplane:
        def hyperplane(w,b,v,x):
            return (v-b-w[0]*x)/w[1]
        #w[0]*x+w[1]*y + b = 0 -> y=-(b+w[0]*x)/w[1]
        plt.plot([xmin,xmax],[hyperplane(w,b,0,xmin),hyperplane(w,b,0,xmax)],'--y')
        plt.plot([xmin,xmax],[hyperplane(w,b,-1,xmin),hyperplane(w,b,-1,xmax)],'k')
        plt.plot([xmin,xmax],[hyperplane(w,b,1,xmin),hyperplane(w,b,1,xmax)],'k')
    plt.xlim(xmin,xmax);plt.ylim(xmin,xmax)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    
if __name__ == "__main__":
    '''
    Run a simple 2D example, comparing soft margin SVM solvution 
    (dual vs regular form) of this code with sklearn solver.
    '''
    np.random.seed(28)
    X,y = Create2DSet(50)
    C=0.1
    svm1 = SVM(C=C,dualForm = True,useCVX = True)
    svm1.fit(X,y)
    print('w: ',svm1.w,' b: ', svm1.b)
    print('accuracy= ',svm1.accuracy(X,y))
    svm2 = SVM(C=C,dualForm = False,useCVX = True)
    svm2.fit(X,y)
    print('w: ',svm2.w,' b: ', svm2.b)
    print('accuracy= ',svm2.accuracy(X,y))
    
    # Verify results with sknlearn solver
    from sklearn.svm import SVC
    svm_check = SVC(C=C,kernel='linear')
    svm_check.fit(X,y)
    b_check = svm_check.intercept_
    w_check = svm_check.coef_
    print('w: ',w_check,' b: ', b_check)
    print('accuracy= ',svm_check.score(X,y))
    #Plot
    import matplotlib.pyplot as plt
    plt.close('all')
    Plot2DSet(X,y,svm1.w,svm1.b,True)
    Plot2DSet(X,y,svm2.w,svm2.b,True)
    plt.show(block=False)