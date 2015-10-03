__author__ = 'YBeer'

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer

classifier_full = KNeighborsClassifier(n_neighbors=3)



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
print len(regression_matrix_indices)

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

"""
full model CV
"""
# CV
cv_n = 4
kf = KFold(dataset.shape[0], n_folds=cv_n, shuffle=True)

print 'start full model evaluation'
for train_index, test_index in kf:
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index].ravel(), y[test_index].ravel()

    # train machine learning
    classifier_full.fit(X_train, y_train)

    # predict
    class_pred = classifier_full.predict_proba(X_test)[:, 1]

    # evaluate
    print 'auc is: ', roc_auc_score(y_test, class_pred)

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

# predict
class_pred = classifier_full.predict_proba(X_test)[:, 1]

submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file['target'] = class_pred
submission_file.to_csv("more_variantes_" + str(uni_thresh) + ".csv")

# date: 13JAN12:00:00:00

# RF: