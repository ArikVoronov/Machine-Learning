'''
K Means classification algorithm
Unsupervised training, but requires k = number of classifications
'''

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

mean1 = [1, 1]
mean2 = [5,5]
cov = [[1,0],[0,2]]
X1 = np.random.multivariate_normal(mean1, cov, 100)
X2 = np.random.multivariate_normal(mean2, cov, 100)

X = np.vstack((X1,X2))

colors = 10*['g','r','c','b','k']

class K_Means:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        # k is the number of classification groups
        self.k = k
        # tolerance for centroid disposition between iterations
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self,data):
        # Create centroid list of length k, intialize with centroids @ first k datapoints
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
        for i in range(self.max_iter):
            print('Iteration {}'.format(i))
            # Create classification dictionary, with k keys (empty at first)
            self.classifications = {}
            for ii in range(self.k):
                self.classifications[ii] = []
            # Calculate the distance between each point of data and every centroid, classify each point by nearest centroid
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            # Create a copy of current centroids for later comparison to tolerance
            prev_centroids = dict(self.centroids)
            # Update centroids as means of newly classified data points
            for c in self.classifications:
                self.centroids[c] = np.average(self.classifications[c],axis=0)
            # Calcuate the maximal total displacement of any of the centroids
            criterion = 0
            for c in self.centroids:
                cent_old = prev_centroids[c]
                cent_new = self.centroids[c]
                criterion = max(np.linalg.norm((cent_new-cent_old)/cent_old*100),criterion)
            print('Criterion:',criterion)
            if criterion < self.tol:
                break     
    
    def predict(self,featureset):
        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
        prediction = distances.index(min(distances))
        return prediction

clf = K_Means(k=2,max_iter=50)
clf.fit(X)

prediction = clf.predict([5,6])
print("Prediction Label:",prediction)


cent = clf.centroids
feats = clf.classifications
for i in clf.centroids:
    plt.scatter(cent[i][0],cent[i][1],s=100,color = 'k',marker = 'x',linewidth = 5)
    for feat in feats[i]:
        plt.scatter(feat[0],feat[1],s=20,color = colors[i])

plt.show()
