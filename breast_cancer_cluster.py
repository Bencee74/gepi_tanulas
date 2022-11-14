# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:54:56 2022

@author: hallgato
"""

import numpy as np;  # Numerical Python library
import matplotlib.pyplot as plt;  # Matlab-like Python module
from sklearn.datasets import load_iris;  # importing data loader
from sklearn.cluster import KMeans; # Class for K-means clustering
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit 
from sklearn.decomposition import PCA; #  Class for Principal Component analysis
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer

digits = load_digits()
bc = load_breast_cancer()

# Default parameters
n_c = 2; # number of clusters

# Enter parameters from consol
user_input = input('Number of clusters [default:2]: ');
if len(user_input) != 0 :
    n_c = np.int8(user_input);
    
# Kmeans clustering
kmeans = KMeans(n_clusters=n_c, random_state=2022);  # instance of KMeans class
kmeans.fit(bc.data);   #  fitting the model to data
bc_labels = kmeans.labels_;  # cluster labels
bc_centers = kmeans.cluster_centers_;  # centroid of clusters
sse = kmeans.inertia_;  # sum of squares of error (within sum of squares)
score = kmeans.score(bc.data);  # negative error
# both sse and score measure the goodness of clustering

# Davies-Bouldin goodness-of-fit
DB = davies_bouldin_score(bc.data,bc_labels);

# Printing the results
print(f'Number of cluster: {n_c}');
print(f'Within SSE: {sse}');
print(f'Davies-Bouldin index: {DB}');

pca = PCA(n_components=2);
pca.fit(bc.data);
bc_pc = pca.transform(bc.data);  #  data coordinates in the PC space
centers_pc = pca.transform(bc_centers);  # the cluster centroids in the PC space


fig = plt.figure(1);
plt.title('Clustering of the Breast cancer data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(bc_pc[:,0],bc_pc[:,1],s=50,c=bc_labels);  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();

# Kmeans clustering with K=2
kmeans = KMeans(n_clusters=2, random_state=2022);  # instance of KMeans class
kmeans.fit(bc.data);   #  fitting the model to data
bc_labels = kmeans.labels_;  # cluster labels
bc_centers = kmeans.cluster_centers_;  # centroid of clusters
distX = kmeans.transform(bc.data);
dist_center = kmeans.transform(bc_centers);

# Visualizing of clustering in the distance space
fig = plt.figure(2);
plt.title('Breast cancer data in the distance space');
plt.xlabel('Cluster 1');
plt.ylabel('Cluster 2');
plt.scatter(distX[:,0],distX[:,1],s=50,c=bc_labels);  # data
plt.scatter(dist_center[:,0],dist_center[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();

# Finding optimal cluster number
Max_K = 31;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(bc.data);
    bc_labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(bc.data,bc_labels);

# Visualization of SSE values    
fig = plt.figure(3);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

# Visualization of DB scores
fig = plt.figure(4);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();