import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self,visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    def fit(self, data):
        self.data = data
        
        opt_dict = {} # { ||w||:[w,b] }

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]
        all_data = []

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # support vectors will fulfill yi(xi*w+b) = 1 (exactly)
        # this is a good criterion for the optimization convergence

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.0005,
                      #self.max_feature_value * 0.0001
                      ]
        # extremely expensive
        b_range_multiple = 1
        # this is a step multiplier only for b
        b_multiple = 10

        latest_optimum = self.max_feature_value*1

        print(latest_optimum)
        b_range = (self.max_feature_value*b_range_multiple)
        #### START OF OPTIMIZATION ####
        for step in step_sizes:
            optimized = False
            for wa in np.arange(latest_optimum,0,-step):
                w = np.array([wa,wa])
                #for b in np.arange(-1*b_range,
                #                   1*b_range,
                #                   step):
##                for b in np.arange(-0.2,0.2,step):
                for transormation in transforms:
                    w_t = w*transormation
                    found_option = True
                    # weakest link in the SVM fundamentally
                    # SMO attempts to fix this a bit
                    # must run check for all values in data
                    # yi(xi*w+b) >= 1
                    values = {1:{},-1:{}}
                    for i in self.data:
                        for ind,xi in enumerate(self.data[i]):
                            yi = i
                            values[yi][np.dot(w_t,xi)] = ([xi])
                    pos = sorted(values[1])
                    neg = sorted(values[-1],reverse=True)
                    #1 = pos[0] +b    #-1 = neg[0] + b
                    b = -1/2*(pos[0]+neg[0])
                    
                    for i in self.data:
                        for xi in self.data[i]:
                            yi = i
                            if not (yi*( np.dot(w_t,xi) + b ) >=1):
                                found_option = False
                    if found_option:
                        #print(w_t,b)
                        #print(yi*(np.dot(w_t,xi+b)))
                        #print(found_option)
                        opt_dict[np.linalg.norm(w_t)] = [w_t,b]
            #print(opt_dict)
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
            b_range = opt_choice[1] + step*2
            print('Optimized a step')
            print('w_opt',self.w,'b_opt',self.b)
            
            
            for i in self.data:
                for ind,xi in enumerate(self.data[i]):
                    yi = i
                    print((yi*( np.dot(self.w,xi) + self.b )))

##            print('b',b_calc)
##            print(pos[0],values[1][pos[0]])
##            print(neg[0],values[-1][neg[0]])
            #b_range = b_calc + step*2


    def predict(self, features):
        # sign( x*w + b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        return classification
    def visualize(self):
        for i in self.data:
            for x in self.data[i]:
                self.ax.scatter(x[0],x[1],s=20,color = self.colors[i])

        def hyperplane(x,w,b,v):
            # w*x+b = v are the svm hyperplanes for v=-1,0,1
            # this function gets "y" values for plotting the hyperplanes
            # assuming you have the x value the eq:
            # w[0]*x + w[1]*y + b = v
            # then: y = (v - b - w[0]*x)/w[1]
            return(v-b-w[0]*x)/w[1]
        datarange = [self.min_feature_value*0.9,self.max_feature_value*1.1]
        x_min = datarange[0]
        x_max = datarange[1]
        for i in [-1,0,1]:
            # hyperplane
            y_min = hyperplane(x_min,self.w,self.b,i)
            y_max = hyperplane(x_max,self.w,self.b,i)
            self.ax.plot([x_min,x_max],[y_min,y_max])
        plt.show()
        

        
        
        
        


data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             1:np.array([[5,1],
                          [6,-1],
                          [7,3],])}

svmach = Support_Vector_Machine()
svmach.fit(data_dict)
svmach.visualize()
