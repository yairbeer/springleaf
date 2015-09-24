__author__ = 'YBeer'

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA

"""
use only columns over threshhold
"""
print 'loading univariante results'
uni_results = pd.read_csv("univar_AUC.csv", index_col=0, names=["index", "AUC"])

uni_thresh = 0.6

regression_matrix_indices = []
for i in range(len(uni_results) - 1):
    if uni_results['AUC'][i] > uni_thresh:
        regression_matrix_indices.append(i)
print 'number if variables ', len(regression_matrix_indices)

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
pca_decomp = PCA(n_components=5)
X = pca_decomp.fit_transform(X)

"""
full model CV, parametric sweep
"""
# CV
cv_n = 4
kf = KFold(dataset.shape[0], n_folds=cv_n, shuffle=True)

print 'start full model evaluation'

classifier_full = GradientBoostingClassifier(loss='deviance')
auc = []
for train_index, test_index in kf:
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index].ravel(), y[test_index].ravel()

    # train machine learning
    classifier_full.fit(X_train, y_train)

    # predict
    class_pred = classifier_full.predict_proba(X_test)[:, 1]

    # evaluate
    auc.append(roc_auc_score(y_test, class_pred))
print 'The auc is ', np.mean(auc)

"""
Evaluate test file
"""
# fitting full model
X_train = X
y_train = y
classifier_full.fit(X_train, y_train)

dataset_test = pd.DataFrame.from_csv("test_col_dummy.csv")
dataset_test = np.array(dataset_test)
X_test = dataset_test[:, regression_matrix_indices]

# preprocess
X_test = imp.transform(X_test)
X_test = scaler.transform(X_test)
X_test = pca_decomp.transform(X_test)

# predict
class_pred = classifier_full.predict_proba(X_test)[:, 1]

submission_file = pd.DataFrame.from_csv("PCA_submission.csv")
submission_file['target'] = class_pred
submission_file.to_csv("first_submission_" + str(uni_thresh) + ".csv")


