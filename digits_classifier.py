# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:57:55 2022

@author: hallgato
"""

import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
from sklearn.linear_model import LogisticRegression; #  importing logistic regression classifier
from sklearn.naive_bayes import GaussianNB; #  importing naive Bayes classifier
from sklearn.model_selection import train_test_split; # importing splitting
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve; #  importing performance metrics
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

data = digits.data
target = digits.target
target_names = digits.target_names

x_train, x_test, y_train, y_test = train_test_split(data,target, test_size=0.3, random_state=2022, 
                                                    shuffle = True)

logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(x_train,y_train);
ypred_logreg = logreg_classifier.predict(x_train);   # spam prediction for train
accuracy_logreg_train = logreg_classifier.score(x_train,y_train);
cm_logreg_train = confusion_matrix(y_train, ypred_logreg); # train confusion matrix
ypred_logreg = logreg_classifier.predict(x_test);   # spam prediction for test
cm_logreg_test = confusion_matrix(y_test, ypred_logreg); # test confusion matrix
yprobab_logreg = logreg_classifier.predict_proba(x_test);  #  prediction probabilities
accuracy_logreg_test = logreg_classifier.score(x_test,y_test);

plot_confusion_matrix(logreg_classifier, x_train, y_train, display_labels = target_names);

plot_confusion_matrix(logreg_classifier, x_test, y_test, display_labels = target_names);


naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(x_train,y_train);
ypred_naive_bayes = naive_bayes_classifier.predict(x_train);  # spam prediction for train
cm_naive_bayes_train = confusion_matrix(y_train, ypred_naive_bayes); # train confusion matrix
ypred_naive_bayes = naive_bayes_classifier.predict(x_test);  # spam prediction
cm_naive_bayes_test = confusion_matrix(y_test, ypred_naive_bayes); # test confusion matrix 
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(x_test);  #  prediction probabilities
accuracy_bayes_train = naive_bayes_classifier.score(x_train, y_train)
accuracy_bayes_test = naive_bayes_classifier.score(x_test, y_test)

# Plotting non-normalized confusion matrix
plot_confusion_matrix(naive_bayes_classifier, x_train, y_train, display_labels = target_names);

plot_confusion_matrix(naive_bayes_classifier, x_test, y_test, display_labels = target_names);


plot_roc_curve(logreg_classifier, x_test, y_test);
plot_roc_curve(naive_bayes_classifier, x_test, y_test);
