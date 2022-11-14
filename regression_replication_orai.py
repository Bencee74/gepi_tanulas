# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:04:58 2022

@author: hallgato
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.utils.random import sample_without_replacement

calif_house = fetch_california_housing().data

n = 20000
b0 = 3
b1 = 2
sig = 1

x = np.random.normal(0,1,n)
eps = np.random.normal(0, sig, n);
y = b0 + b1*x + eps


rep = 20
sample_size = 1000
reg = LinearRegression();  # instance of the LinearRegression class
b0hat = [];  #  list for intercept
b1hat = [];  # list for slope
score = [];  # list for R-squares

for i in range(rep):
    # random sampling from dataset
    index = sample_without_replacement(n_population = n, n_samples = sample_size);
    x_sample = x[index];
    y_sample = y[index];
    X_sample = x_sample.reshape(1, -1).T;
    reg.fit(X_sample,y_sample);
    b0hat.append(reg.intercept_);
    b1hat.append(reg.coef_);
    score.append(reg.score(X_sample,y_sample));
    
b0hat_mean = np.mean(b0hat);
b0hat_std = np.std(b0hat);
b1hat_mean = np.mean(b1hat);
b1hat_std = np.std(b1hat);
score_mean = np.mean(score);
score_std = np.std(score);

print(f'Mean slope:{b1hat_mean:6.4f} (True slope:{b1}) with standard deviation {b1hat_std:6.4f}');
print(f'Mean intercept:{b0hat_mean:6.4f} (True intercept:{b0}) with standard deviation {b0hat_std:6.4f}');
print(f'Mean of R-square for goodness of fit:{score_mean:6.4f} (standard deviation: {score_std:6.4f})');
 

