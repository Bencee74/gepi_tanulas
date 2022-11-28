# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:27:52 2022

@author: hallgato
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
import seaborn as sns;  # importing statistical data visualization
from sklearn.datasets import load_digits; # importing Iris dataset

d = np.random.normal(0,1,(1000,2))
    
df = pd.DataFrame(d, columns=['var1','var2'])

digits = load_digits(as_frame=True)
n = digits.data.shape[0]
p = digits.data.shape[1]
k = digits.target_names.shape[0]

digits_by_target = digits.frame.groupby(by='target');  # grouping by target

# Basic descriptive stats
mean_by_target = digits_by_target.mean();  # mean
std_by_target = digits_by_target.std();  # standard deviation
corr_by_target = digits_by_target.corr();  #  correlations
desc_stat_by_target = digits_by_target.describe();  # desc stat with quantiles

plt.figure(1);
pd.plotting.andrews_curves(digits.frame,class_column='target');
plt.show();

plt.figure(2);
pd.plotting.parallel_coordinates(digits.frame,class_column='target');
plt.show();

pd.plotting.scatter_matrix(digits.data);
