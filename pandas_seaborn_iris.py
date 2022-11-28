# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:33:31 2020

Task: Statistical analysis and visualization of Iris dataset
using pandas and seaborn

Python tools    
Libraries: numpy, sklearn, pandas, seaborn
Modules: pyplot, plotting
Classes:  
Functions: 

@author: MĂĄrton IspĂĄny
"""

import numpy as np;  # importing numerical computing package
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
import pandas as pd;  # importing pandas data analysis tool
import seaborn as sns;  # importing statistical data visualization
from sklearn.datasets import load_iris; # importing Iris dataset

# Loading dataset as frame
iris = load_iris(as_frame=True);
n = iris.data.shape[0]; # number of records
p = iris.data.shape[1]; # number of attributes
k = iris.target_names.shape[0]; # number of target values

# Basic pandas

iris_by_target = iris.frame.groupby(by='target');  # grouping by target
# Exporting a target group from the grouped dataframe
for i in range(k):
    if iris.target_names[i]=='setosa': setosa_ind=i;
iris_setosa = iris_by_target.get_group(setosa_ind);
iris_set = iris_setosa.drop(columns=['target']);
first = 5;
# Printing the first observations by target
for target_value, group in iris_by_target:
   print ('Target: ',iris.target_names[target_value]);
   print (group[0:first]);

# Basic descriptive stats
mean_by_target = iris_by_target.mean();  # mean
std_by_target = iris_by_target.std();  # standard deviation
corr_by_target = iris_by_target.corr();  #  correlations
desc_stat_by_target = iris_by_target.describe();  # desc stat with quantiles

# Printing the results
print(mean_by_target);

# Plotting using multidimensional tools of pandas
# Andrews curves
plt.figure(1);
pd.plotting.andrews_curves(iris.frame,class_column='target',color=['blue','green','red']);
plt.show();
# Parallel axis
plt.figure(2);
pd.plotting.parallel_coordinates(iris.frame,class_column='target',color=['blue','green','red']);
plt.show();

# Scatter matrix
pd.plotting.scatter_matrix(iris.data);

# Basic seaborn                         

# Loading seaborn's default theme and color palette
sns.set(); 

# Plotting two attributes
colors = ['blue','red','green'];
sns.relplot(data=iris.frame, x='sepal length (cm)', y='petal length (cm)', 
            hue='target', palette=colors);
# Default axis
x_axis = 0;  # x axis attribute (0..3)
y_axis = 1;  # y axis attribute (0..3)
# Enter axis from consol
user_input = input('X axis [0..3, default:0]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=3 :
    x_axis = np.int8(user_input);
user_input = input('Y axis [0..3, default:1]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=3 :
    y_axis = np.int8(user_input); 
sns.relplot(data=iris.frame, x=iris.feature_names[x_axis], 
            y=iris.feature_names[y_axis], hue='target', palette=colors);
plt.figure(6);    
sns.scatterplot(data=iris.frame, x=iris.feature_names[x_axis], 
                        y=iris.feature_names[y_axis], hue='target', palette=colors);
plt.show();

plt.figure();
sns.displot(data=iris.frame, x=iris.feature_names[x_axis], kde=True)
plt.show();

plt.figure();
sns.histplot(data=iris.frame, x=iris.feature_names[x_axis], hue='target');
plt.show();

plt.figure();
sns.boxplot(x=iris.frame['target'], 
            y=iris.frame['sepal length (cm)']);
plt.show();

plt.figure();
sns.pairplot(data=iris.frame,hue='target');
plt.show();

plt.figure();
sns.heatmap(iris.data)
plt.show();