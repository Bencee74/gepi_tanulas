# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np;
from urllib.request import urlopen;
from sklearn import model_selection as ms;
from matplotlib import pyplot as plt;
import pandas as pd;
import seaborn as sns;
from sklearn import tree;
from sklearn.model_selection import train_test_split;
from sklearn.tree import DecisionTreeClassifier, plot_tree;
from sklearn.linear_model import LogisticRegression;
from sklearn.model_selection import train_test_split;
from sklearn.neural_network import MLPClassifier;
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc, plot_roc_curve, davies_bouldin_score;
from sklearn.cluster import KMeans;
from sklearn.decomposition import PCA;

# 1. feladat
url = 'https://arato.inf.unideb.hu/ispany.marton/MachineLearning/Datasets/labor_exercise_wednesday2.csv';
raw_data = urlopen(url);
data = np.loadtxt(raw_data, delimiter=',');
del raw_data;

# 2. feladat
# Elemek lekérdezése: df['Var1'][8];
df = pd.DataFrame(data, columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5',
                                 'Var6', 'Var7', 'Var8', 'Var9', 'Var10',
                                 'Target']);
data_by_target = df.groupby(by='Target');


mean_by_target = data_by_target.mean();
std_by_target = data_by_target.std();

# 3. feladat
# Andrews-ábra
plt.figure(1);
pd.plotting.andrews_curves(frame=df, class_column='Target', color=['red', 'green']);
plt.show();

#Párhuzamos tengelyek
plt.figure(2);
pd.plotting.parallel_coordinates(df, class_column='Target', color=['red', 'green']);
plt.show();

#Matrix-plot
pd.plotting.scatter_matrix(df);

#Seaborn matrix
plt.figure(3);
sns.pairplot(data=df,hue='Target');
plt.show();

# 4. feladat
# Defining input and target variables
X = data[:,0:10];
y = data[:,10];

X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.3, 
                                shuffle = True, random_state=2022);

# 5. feladat
# Initialize the decision tree object
# Criterium: gini, Depth: 4
crit = 'gini';
depth =4;
# Instance of decision tree class
class_tree = tree.DecisionTreeClassifier(criterion=crit,max_depth=depth);
# Fitting decision tree on training dataset
class_tree.fit(X_train, y_train);
score_train_tree = class_tree.score(X_train, y_train); # Goodness of tree on training dataset
score_test_tree = class_tree.score(X_test, y_test); # Goodness of tree on test dataset

# Logisztikus regresszió liblinearis solverrel
logreg_classifier = LogisticRegression(solver = 'liblinear');
logreg_classifier.fit(X_train,y_train);
score_train_logreg = logreg_classifier.score(X_train,y_train);
score_test_logreg = logreg_classifier.score(X_test,y_test);
ypred_logreg = logreg_classifier.predict(X_test);
yprobab_logreg = logreg_classifier.predict_proba(X_test);

#Neurális háló
neural_classifier = MLPClassifier(hidden_layer_sizes=(1,2),activation='logistic',max_iter=1000);  #  number of hidden neurons: 5
neural_classifier.fit(X_train,y_train);
score_train_neural = neural_classifier.score(X_train,y_train);  #  goodness of fit
score_test_neural = neural_classifier.score(X_test,y_test);  #  goodness of fit
ypred_neural = neural_classifier.predict(X_test);   # spam prediction
yprobab_neural = neural_classifier.predict_proba(X_test);  #  prediction probabilities

print(f'Test score of tree in %: {score_test_tree*100}');
print(f'Test score of logreg in %: {score_test_logreg*100}'); 
print(f'Test score of neural in %: {score_test_neural*100}');

# 6. feladat
ypred_tree = class_tree.predict(X_test);
yprobab_tree = class_tree.predict_proba(X_test);
cm = confusion_matrix(y_test, ypred_tree);

fpr_tree, tpr_tree, _ = roc_curve(y_test, yprobab_tree[:,0], pos_label=0);
roc_auc_tree = auc(fpr_tree, tpr_tree);
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,0], pos_label=0);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);
fpr_neural, tpr_neural, _ = roc_curve(y_test, yprobab_neural[:,0], pos_label=0);
roc_auc_neural = auc(fpr_neural, tpr_neural);

plt.figure(4);
lw = 2;
plt.plot(fpr_tree, tpr_tree, color='red',
         lw=lw, label='Decision Tree (AUC = %0.2f)' % roc_auc_tree);
plt.plot(fpr_logreg, tpr_logreg, color='green',
         lw=lw, label='Logistic Regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_neural, tpr_neural, color='blue',
         lw=lw, label='Neural Network (AUC = %0.2f)' % roc_auc_neural);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();

# 7. feladat
# Default parameters
n_c = 3; # number of clusters

# Enter parameters from consol
    
kmeans = KMeans(n_clusters=n_c, random_state=2022);  # instance of KMeans class
kmeans.fit(X_train);   #  fitting the model to data
labels = kmeans.labels_;  # cluster labels
centers = kmeans.cluster_centers_;  # centroid of clusters
sse = kmeans.inertia_;  # sum of squares of error (within sum of squares)
score = kmeans.score(X_train);  # negative error
# both sse and score measure the goodness of clustering

# Davies-Bouldin goodness-of-fit
DB = davies_bouldin_score(X_train, labels);

# Printing the results
print(f'Number of cluster: {n_c}');
print(f'Within SSE: {sse}');
print(f'Davies-Bouldin index: {DB}');

Max_K = 30;  # maximum cluster number
SSE = np.zeros((Max_K-2));  #  array for sum of squares errors
DB = np.zeros((Max_K-2));  # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2022);
    kmeans.fit(X_test);
    iris_labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(X_train, labels);
    

pca = PCA(n_components=2);
pca.fit(X_train);
pc = pca.transform(X_train);  #  data coordinates in the PC space
centers_pc = pca.transform(X_train);  # the cluster centroids in the PC space

fig = plt.figure(4);
plt.title('Clustering of the data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(pc[:,0],pc[:,1],s=50,c=labels);  # data
plt.scatter(centers_pc[:,0],centers_pc[:,1],s=200,c='red',marker='X');  # centroids
plt.legend();
plt.show();


