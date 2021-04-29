# Check out Sequential Minimal Optimization
import numpy as np
import sys
def QuadOptimization(P,q,A,b,G,h):
    '''
    Quadaratic Programming OPTIMIZATION
    min 1/2 x.T*P*x + q.T*x
    s.t. Gx <= h (-a<=0)
        Ax <= b (sum(y*a)=0)
        
    Lagrangian:
    L(x,lam,mu) = q.Tx +1/2*x.T*P*x + lam.T(Ax-b) + mu.T(Gx-h) = 0
    '''
    def ValidCheck(G,h,x):
        '''
        Helper function, checks if inequalities are satisfied for input data point x
        '''
        tol=1e-5
        remainder = np.dot(G,x)-h
        valids = remainder<=0
        constraintsSatisfied = valids.all()
        unsatisfied = np.argwhere((1-valids).squeeze())
        if unsatisfied.size >0:
            dt = np.dot(G[unsatisfied.squeeze(),:],x)-h[unsatisfied.squeeze(),:]
            ind = np.argsort(dt,0)
            unsatisfied = unsatisfied[ind[::-1]] # Largest remainder first
        return unsatisfied
    
    if A.size==0 and b.size==0 and G.size==0 and h.size==0: # no constraints case
        #xOpt = np.dot(np.linalg.inv(P),-q)
        xOpt = np.linalg.solve(P, -q)
        return xOpt
    elif G.size==0 and h.size==0:
        # Only eq constraints, direct solution
        xDim = P.shape[0]
        M1 = np.hstack([P,A.T])
        M2 = np.hstack([A,np.zeros([A.shape[0],A.shape[0]+P.shape[1]-A.shape[1]])])
        M = np.vstack([M1,M2])
        rhs = np.vstack([-q,b])
        opt = np.linalg.solve(M, rhs)
        xOpt = np.round(opt[:xDim],10)
        return xOpt
        
        
    else: 
        # ineq (with or without eq) constraints
        xDim = P.shape[0]
        activeInEq = np.arange(min(xDim,G.shape[0]))#Tracks currently active ineq constraints
        itr=0
        while True: # Run this loop until a VALID optimal point is obtained or reached maximum iterations
            if itr>=1e4:
                print("REACHED MAX ITERATIONS!")
                break
            itr+=1
            # Formulate LHS: [P AG.T;AG 0]
            if activeInEq.size==0:
                # No ineq
                if A.size==0:
                    M = P
                    rhs = -q
                else:
                    M1 = np.hstack([P,A.T])
                    M2 = np.hstack([A,np.zeros([A.shape[0],A.shape[0]+P.shape[1]-A.shape[1]])])
                    M = np.vstack([M1,M2])
                    rhs = np.vstack([-q,b])
            else:
                Gc = G[activeInEq,:]
                hc = h[activeInEq,:]
                if A.size==0:
                    AG = Gc
                elif Gc.size==0:
                    AG = A
                else:
                    AG = np.vstack([A,Gc])
                M1 = np.hstack([P,AG.T])
                M2 = np.hstack([AG,np.zeros([AG.shape[0],AG.shape[0]+P.shape[1]-AG.shape[1]])])
                M = np.vstack([M1,M2])

                # Formulate RHS [-q;b;h]
                if b.size==0:
                    rhs = np.vstack([-q,hc])
                elif hc.size==0:
                    rhs = np.vstack([-q,b])
                else:
                    rhs = np.vstack([-q,b,hc])
            # Calculate optimal values for features and Lagrange multpiliers
            opt = np.linalg.solve(M, rhs)
            xOpt = np.round(opt[:xDim],10)
            if activeInEq.size>0:
                lamOpt = opt[-activeInEq.size:] # <- these are the LM used to check if the inequalities are active/can be droppe           
            else: # only equalities!
                lamOpt = np.array([])
            negativeLams = np.argwhere(lamOpt.squeeze()<=0)
            unsatisfied = ValidCheck(G,h,xOpt) # check which constraints (of the original FULL ineq) are satisfied
            if unsatisfied.size ==0:
                if len(activeInEq)==0: # Case where the optimal point doesn't activate ANY constraints
                    break
                if len(negativeLams)>0:
                    mask = np.ones_like(activeInEq,dtype=bool)
                    mask[negativeLams]=False
                    activeInEq = activeInEq[mask]
                    continue
                else: # Case where all active constraints have positive lams, and ALL constraints are satisfied, that's the optimum
                    break
            else: # if some constraints are not satisfied, activate them for the next iteration
                if len(negativeLams)>0:
                    mask = np.ones_like(activeInEq,dtype=bool)
                    mask[negativeLams]=False
                    activeInEq = activeInEq[mask]
                    continue
                if activeInEq.size>=(xDim-A.shape[0]):
                    activeInEq = np.delete(activeInEq,0)
                activeInEq = np.append(activeInEq,unsatisfied[0])
        return xOpt

    
if __name__=="__main__":
    # Simple example in 2D
    P = np.array([[4,1],[1,4]])
    q = np.array([2,1])[:,None]
    A = np.array([1,-1])[None,:]
    b = np.array([0])[None,:]
    G = np.array([-1,-1])[None,:]
    h = np.array([-1])[None,:]
    xOpt = QuadOptimization(P,q,A,b,G,h)
    x = np.linspace(-2,2,1001)
    y = np.linspace(-3,3,1001)
    X,Y = np.meshgrid(x,y)
    Z = 1/2*(P[0][0]*X**2 + P[0][1]*X*Y + P[1][0]*Y*X + P[1][1]*Y**2 )+q[0]*X + q[1]*Y
    def F(xx):
        f = np.dot(xx.T,np.dot(P,xx)) + np.dot(q,xx)
        return f
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    plt.contour(X,Y,Z)
    plt.scatter(xOpt[0],xOpt[1])
    plt.plot(x,(b[0]-A[0][0]*x)/A[0][1])
    plt.plot(x,(h[0]-G[0][0]*x)/G[0][1])
