import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

__author__ = 'YBeer'

"""
use only columns over threshhold
"""
print 'loading univariante results'
uni_results = pd.read_csv("univar_AUC.csv", index_col=0, names=["index", "AUC"])

# print regression_matrix_indices
print 'loading dataset'
dataset = pd.DataFrame.from_csv("train_col_dummy.csv")
rows = random.sample(dataset.index, 1000)
dataset = dataset.ix[rows]

print 'changing to array'
dataset = np.array(dataset)

uni_thresh = 0.62
print 'threshold is ', uni_thresh
regression_matrix_indices = []
for i in range(len(uni_results) - 1):
    if uni_results['AUC'][i] > uni_thresh:
        regression_matrix_indices.append(i)
print len(regression_matrix_indices)

X = dataset[:, regression_matrix_indices]
y = dataset[:, -1].ravel()

# impotate
print 'impotating'
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
imp.fit(X)
X = imp.transform(X)

# standardizing results
print 'standardizing results'
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# PCA
print 'PCA results'
pca_decomp = PCA(n_components=5)
X = pca_decomp.fit_transform(X)
print X.shape

"""
split data
"""
col_1 = 1
col_2 = 2
split_true = []
split_false = []
for i in range(X.shape[0]):
    if y[i]:
        split_true.append([X[i, col_1], X[i, col_2]])
    else:
        split_false.append([X[i, col_1], X[i, col_2]])
split_true = np.array(split_true)
split_false = np.array(split_false)
plt.plot(split_true[:, 0], split_true[:, 1], 'go', split_false[:, 0], split_false[:, 1], 'ro')
plt.xlim((-20, 20))
plt.xlim((-20, 20))
plt.show()
