import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math

def euclidean_distance(feat_one, feat_two):

    squared_distance = 0

    #Assuming correct input to the function where the lengths of two features are the same

    for i in range(len(feat_one)):
        squared_distance += (feat_one[i] - feat_two[i])**2

    ed = math.sqrt(squared_distance)

    return ed;

def customKMeans(data, k, tolerance, max_iterations):
    centroids = {}
    
    #initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
    for i in range(k):
        centroids[i] = data[i]
    
    #begin iterations
    for i in range(max_iterations):
        clusters = {}
        for j in range(k):
            clusters[j] = []
    
        #find the distance between the point and cluster; choose the nearest centroid
        for features in data:
            dist = []
            for j in range(k):
                dist.append(euclidean_distance(centroids[j], features))
            clusterKey = dist.index(min(dist))
            clusters[clusterKey].append(features)
    
        previous = dict(centroids)
    
        #average the cluster datapoints to re-calculate the centroids
        for clusterKey in range(k):
            centroids[clusterKey] = np.average(clusters[clusterKey], axis = 0)
    
        displacement = 0
        isOptimal = True
        for centroidKey in range(k):
            prev = previous[centroidKey]
            curr = centroids[centroidKey]
            displacement = euclidean_distance(curr, prev)
            if displacement > tolerance:
                isOptimal = False
        
        #break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
        if isOptimal:
            break
    
    print("Centroids:: ", centroids)
    return centroids, clusters 

df = pd.read_csv("./data/ipl.csv")
df = df[['one', 'two']]

data = df.values #returns a numpy array
k = 3
tolerance = 0.05
max_iterations = 500

centroids, clusters = customKMeans(data, k, tolerance, max_iterations)

# Plotting starts here
colors = k*["r", "g", "c", "b", "k"]

for centroidKey in range(k):
    plt.scatter(centroids[centroidKey][0], centroids[centroidKey][1], marker = "x", color = colors[centroidKey])

for clusterKey in range(k):
    color = colors[clusterKey]
    for features in clusters[clusterKey]:
        plt.scatter(features[0], features[1], color = color)

plt.show()


