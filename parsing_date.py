import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold

__author__ = 'yaia'

# get target
y = pd.Series.from_csv("target.csv")
y = np.array(y)[1:]
n = y.shape[0]

# get columns
X = pd.DataFrame.from_csv("VAR_0204.csv")
X = np.array(X)

# split datetime
X_split = np.ones((n, 4)) * (-1)
X_split = X_split.astype('str')
for i in range(n):
    if str(X[i, 0]) != 'nan':
        X_split[i, 0] = X[i, 0][:2]
        X_split[i, 1] = X[i, 0][2:5]
        X_split[i, 2] = X[i, 0][5:7]
        X_split[i, 3] = X[i, 0][8:10]

# convert to DF
X_cols = ['day', 'month', 'year', 'hour']
X_split = pd.DataFrame(X_split, columns=X_cols)

# get dummy variables
dummies = []
for col in X_cols:
    dummies.append(pd.get_dummies(X_split[col]))

# Concate data and remove duplicates
X_split = pd.concat(dummies, axis=1)
X_split = X_split.T.drop_duplicates().T
X_split = np.array(X_split)

# CV
ml = GradientBoostingClassifier(loss='deviance', learning_rate=0.2, n_estimators=150, max_depth=3, max_features=None)
cv_n = 4
kf = KFold(n, n_folds=cv_n, shuffle=True)

print 'start dummy evaluation'
auc = []
for train_index, test_index in kf:
    X_train, X_test = X_split[train_index, :], X_split[test_index, :]
    y_train, y_test = y[train_index].ravel(), y[test_index]
    y_test = y_test.astype('float64')

    # train machine learning
    ml.fit(X_train, y_train)

    # predict
    class_pred = ml.predict_proba(X_test)[:, 1]

    # evaluate
    auc.append(roc_auc_score(y_test, class_pred))
print 'auc is: ', np.mean(auc)
