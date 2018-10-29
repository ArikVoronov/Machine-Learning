import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
import random


def k_nearest_neighbors(data,predict,k=3):
    # Each label should have at least one chance for a vote
    if len(data) >= k:
        warnings.warn('bad K')
    distances = []
    # Calculate distances of all points in the data from the prediction point
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    # Create a list of labels of k nearest datapoints
    votes = [ i[1] for i in sorted(distances)[:k] ]
    # Find the most common 'vote'
    vote_result = Counter(votes).most_common(1)[0][0] 
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence



'''
K nearest neightbours classification algorithm
Supervised training
'''
df = pd.read_csv('breast-cancer-wisconsin.data.txt')


# Make all missing datapoints far away and irrelevant 
df.replace('?',-99999, inplace=True)
# id is irrelevant for predictions
df.drop(['id'],1,inplace=True)


# Rows of df converted to lists
full_data = df.astype(float).values.tolist()

# Split data

# Data is shuffled (mostly for reruns of the algorithm, making sure the accuracy isn't arbitrary)
random.shuffle(full_data)
test_size = 0.4
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# Populate dictionaries, put features in corresponding labels
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

'''
Run the algorithm , brute force:

For each member of the test set, calculate prediction of label using the training set
Compare the prediction to the actual label (the feature key in the dictionary)
If the predicition is the same as the label, add to correct counter
Accuracy is the ratio between correct and total predictions

'''
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set,data,200)
        if group == vote:
            correct+=1
        else:
            print(confidence)
        total+=1

print('Accuracy:', correct/total)
