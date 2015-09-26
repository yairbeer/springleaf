import pandas as pd
import numpy as np
from glob import glob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from datetime import date

__author__ = 'yaia'

# month dictionary
dictionary = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10,
              'NOV': 11, 'DEC': 12, }

# get target
y = pd.Series.from_csv("target.csv")
y = np.array(y)[1:]
n = y.shape[0]

# get columns
files = glob('VAR_*')
for file_name in files:
    X = pd.DataFrame.from_csv(file_name)
    X = np.array(X)

    # split datetime
    X_split = np.ones((n, 5)) * (-1)
    X_split = X_split.astype('str')
    for i in range(n):
        if str(X[i, 0]) != 'nan':
            cur_datetime = X[i, 0]
            cur_date = date(2000 + int(cur_datetime[5:7]), dictionary[cur_datetime[2:5]], int(cur_datetime[:2]))
            X_split[i, 0] = cur_date.year
            X_split[i, 1] = cur_date.month
            X_split[i, 2] = cur_date.day
            X_split[i, 3] = cur_datetime[8:10]
            X_split[i, 4] = cur_date.weekday()

    # convert to DF
    X_cols = ['day', 'month', 'year', 'hour', 'weekday']
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
    ml = GradientBoostingClassifier(loss='deviance', learning_rate=0.2, n_estimators=150, max_depth=3,
                                    max_features=None)
    cv_n = 4
    kf = KFold(n, n_folds=cv_n, shuffle=True)

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
    print file_name, '\'s auc is: ', np.mean(auc)

# try to use hour as an int
# add weekdays

# without weekdays
# VAR_0073.csv 's auc is:  0.632563213696
# VAR_0075.csv 's auc is:  0.586792449527
# VAR_0156.csv 's auc is:  0.520748159105
# VAR_0157.csv 's auc is:  0.504222153918
# VAR_0158.csv 's auc is:  0.503908630126
# VAR_0159.csv 's auc is:  0.520651153676
# VAR_0166.csv 's auc is:  0.542473141234
# VAR_0167.csv 's auc is:  0.511193353482
# VAR_0168.csv 's auc is:  0.525478066271
# VAR_0169.csv 's auc is:  0.542489900662
# VAR_0176.csv 's auc is:  0.554388644498
# VAR_0177.csv 's auc is:  0.515065511893
# VAR_0178.csv 's auc is:  0.528280723983
# VAR_0179.csv 's auc is:  0.554406420132
# VAR_0204.csv 's auc is:  0.522435927466
# VAR_0217.csv 's auc is:  0.528191615785
