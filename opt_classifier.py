import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer

__author__ = 'YBeer'

"""
use only columns over threshhold
"""
print 'loading univariante results'
uni_results = pd.read_csv("univar_AUC.csv", index_col=0, names=["index", "AUC"])

uni_thresh = 0.3
print 'threshold is ', uni_thresh
regression_matrix_indices = []
for i in range(len(uni_results) - 1):
    if uni_results['AUC'][i] > uni_thresh:
        regression_matrix_indices.append(i)
print len(regression_matrix_indices)
# print regression_matrix_indices

print 'loading dataset'
dataset = pd.DataFrame.from_csv("train_col_dummy.csv")

print 'changing to array'
dataset = np.array(dataset)

X = dataset[:, regression_matrix_indices]
y = dataset[:, -1]

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
# print 'PCA results'
# pca_decomp = PCA(n_components=100)
# X = pca_decomp.fit_transform(X)
# print X.shape

"""
full model CV
"""
# CV
cv_n = 4
kf = KFold(dataset.shape[0], n_folds=cv_n, shuffle=True)
print 'start full model evaluation'

item_list = [0.05, 0.1, 0.25, 0.5, 0.75]
for item in item_list:
    print item
    classifier = RandomForestClassifier(max_depth=12, n_estimators=30, max_features=0.25)
    auc = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index].ravel(), y[test_index].ravel()

        # train machine learning
        classifier.fit(X_train, y_train)

        # predict
        class_pred = classifier.predict_proba(X_test)[:, 1]

        # evaluate
        auc.append(roc_auc_score(y_test, class_pred))
    print i, ' auc is: ', np.mean(auc)
