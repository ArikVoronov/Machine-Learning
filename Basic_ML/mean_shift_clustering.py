import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

##X = np.array([[1,2],
##              [1.5,1.8],
##              [5,8],
##              [8,8],
##              [1,0.6],
##              [9,11],
##              [8,2],
##              [10,2],
##              [9,3]])

mean1 = [11, 10]
mean2 = [15,15]
mean3 = [19,11]
cov = [[1,0],[0,1]]
X1 = np.random.multivariate_normal(mean1, cov, 10)
X2 = np.random.multivariate_normal(mean2, cov, 10)
X3 = np.random.multivariate_normal(mean3, cov, 10)
X = np.vstack((X1,X2,X3))

colors = 10*['r','g','b','c','orange']

class Mean_Shift:
    def __init__(self, radius = None, radius_norm_step = 100,tol = 1e-2):
##        self.radius = radius
##        self.radius_norm_step = radius_norm_step
        self.tol = tol
        
    def fit(self,data):
        
        self.all_data_centroid = np.average(data,axis = 0)
        norm_v=[]
        for d in data:
            norm_v.append(np.linalg.norm( d-self.all_data_centroid))
        self.norm_v = norm_v
        all_data_norm = np.std(norm_v)
        print('norms',max(norm_v),np.average(norm_v),np.std(norm_v))
        self.all_data_norm = all_data_norm

        # Radius 
##        if self.radius == None:
##            self.radius = all_data_norm / self.radius_norm_step
##            print(self.radius)

        # Start with all data points as centroids
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                # Centroids are calculated by a weighted average
                # The weights are a gaussian function (inverse to distance from centroid)
                # W = exp(-distance^2/std^2)
                # where STD is calculated above - the standard deviation of the distances of all data points from
                # the common centroid
                weights = np.zeros(len(data))
                to_sum = 0
                for ii,featureset in enumerate(data):
                    distance = np.linalg.norm(featureset-centroid)
                    crit = distance/self.all_data_norm
                    weights[ii] = np.exp(-crit**2)
                    to_sum += weights[ii]*featureset
                    
                new_centroid = to_sum/np.sum(weights)
                new_centroids.append(tuple(new_centroid))
                
            
            # Find and remove close centroids:
            uniques = sorted(list(set(new_centroids)))
            to_pop = []
            for ind,i in enumerate(uniques):
                for ii in uniques[(ind+1):]:
                    criterion = np.linalg.norm(np.array(i)-np.array(ii))
                    if criterion <= self.all_data_norm:
                        to_pop.append(ii)
                        break
##                    if i == ii:
##                        pass
##                    elif criterion <= self.all_data_norm:
##                        to_pop.append(ii)
##                        break
            
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass


            # Check for convergence
            prev_centroids = dict(centroids)
            centroids = {}
            
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])
            optimized = True

            for i in centroids:
                crit = np.linalg.norm(centroids[i]-prev_centroids[i])
                print('crit',crit)
                if crit > self.tol*self.all_data_norm:
                    optimized = False
                if not optimized:
                    break
            
            if optimized:
                break

        self.centroids = centroids
        

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self,featureset):
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            return classification

clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids


#plt.scatter(X[:,0],X[:,1], s=150)
##print(centroids)

print('centroids',len(centroids),centroids)
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker='o',color=color,s=150)
for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],color = 'k', s=100,marker = 'x')


x0,y0 = clf.all_data_centroid
rad1 = np.average(clf.norm_v)
rad_std = np.std(clf.norm_v)
x1, x2 = np.meshgrid(np.linspace(5,25,50), np.linspace(5,25,50))

plt.contour(x1, x2, (x1-x0)**2+(x2-y0)**2-rad1**2 , [0.0], colors='r', linewidths=1, origin='lower')
plt.contour(x1, x2, (x1-x0)**2+(x2-y0)**2-(rad1-rad_std)**2 , [0.0], colors='c', linewidths=1, origin='lower')
plt.contour(x1, x2, (x1-x0)**2+(x2-y0)**2-(rad1+rad_std)**2 , [0.0], colors='c', linewidths=1, origin='lower')
plt.show()

                    
