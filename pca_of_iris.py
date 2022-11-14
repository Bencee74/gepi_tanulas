# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23:34:18 2020

Pricipal Component Analysis (PCA) of Iris data  

@author: MĂĄrton IspĂĄny
"""

import sklearn.datasets as ds; # importing datasets
import sklearn.decomposition as dc;  # importing dimension reduction
import numpy as np;  # importing numerical computing package
import matplotlib.pyplot as plt;  # importing MATLAB-like plotting framework
import matplotlib.colors as col;  # importing coloring tools from MatPlotLib
import seaborn as sns;  # importing the Seaborn library
from sklearn.feature_selection import SelectKBest;
 
# loading dataset
iris = ds.load_iris();
bc = ds.load_breast_cancer()





# Scatterplot for two input attributes
x_axis = 1;  # x axis attribute (0,1,2,3)
y_axis = 2;  # y axis attribute (0,1,2,3)
colors = ['blue','red','green']; # colors for target values: setosa blue, versicolor red, virginica green
n = iris.data.shape[0]; # number of records
p = iris.data.shape[1]; # number of attributes
k = iris.target_names.shape[0]; # number of target classes

bc_n = bc.data.shape[0]
bc_p = bc.data.shape[1]
bc_k = bc.target_names.shape[0]
feature_selection = SelectKBest(k=3)
feature_selection.fit(bc.data, bc.target)
scores = feature_selection.scores_;
features = feature_selection.transform(bc.data)
mask = feature_selection.get_support()
feature_indices = []

for i in range(bc_p):
    if mask[i] == True : feature_indices.append(i)

x_axis, y_axis, z_axis = feature_indices
 
fig = plt.figure(1);
plt.title('Scatterplot for iris dataset');
plt.xlabel(iris.feature_names[x_axis]);
plt.ylabel(iris.feature_names[y_axis]);
plt.scatter(iris.data[:,x_axis],iris.data[:,y_axis],s=50,c=iris.target,cmap=col.ListedColormap(colors));
plt.show();

# Matrix scatterplot of Iris
sns.set(style="ticks");
df = sns.load_dataset("iris");
sns.pairplot(df, hue="species");

# Full PCA using scikit-learn
pca = dc.PCA();
pca.fit(iris.data);

# Visualizing the variance ratio which measures the importance of PCs
fig = plt.figure(3);
plt.title('Explained variance ratio plot');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 

# Visualizing the cumulative ratio which measures the impact of first n PCs
fig = plt.figure(4);
plt.title('Cumulative explained variance ratio plot');
cum_var_ratio = np.cumsum(var_ratio);
x_pos = np.arange(len(cum_var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,cum_var_ratio, align='center', alpha=0.5);
plt.show(); 

# PCA with limited components
pca = dc.PCA(n_components=2);
pca.fit(iris.data);
iris_pc = pca.transform(iris.data);
class_mean = np.zeros((k,p));
for i in range(k):
    class_ind = [iris.target==i][0].astype(int);
    class_mean[i,:] = np.average(iris.data, axis=0, weights=class_ind);
PC_class_mean = pca.transform(class_mean);    
full_mean = np.reshape(pca.mean_,(1,4));
PC_mean = pca.transform(full_mean);

fig = plt.figure(5);
plt.title('Dimension reduction of the Iris data by PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(iris_pc[:,0],iris_pc[:,1],s=50,c=iris.target,
            cmap=col.ListedColormap(colors),label='Datapoints');
plt.scatter(PC_class_mean[:,0],PC_class_mean[:,1],s=50,marker='P',
            c=np.arange(k),cmap=col.ListedColormap(colors),label='Class means');
plt.scatter(PC_mean[:,0],PC_mean[:,1],s=50,c='black',marker='X',label='Overall mean');
plt.legend();
plt.show();