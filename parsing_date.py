import pandas as pd
import numpy as np
from glob import glob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from datetime import date

__author__ = 'YBeer'

# month dictionary
dictionary = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10,
              'NOV': 11, 'DEC': 12, }

# get target
y = pd.Series.from_csv("target.csv")
y = np.array(y)[1:]
n = y.shape[0]

ml = GradientBoostingClassifier()
cv_n = 4
kf = KFold(n, n_folds=cv_n, shuffle=True)

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
    X_split_no_weekday = pd.DataFrame(X_split[:, :-1], columns=X_cols[:-1])
    X_split = pd.DataFrame(X_split, columns=X_cols)

    print file_name
    print 'with weekday'
    # get dummy variables
    dummies = []
    for col in X_cols:
        dummies.append(pd.get_dummies(X_split[col]))

    # Concate data and remove duplicates
    X_split = pd.concat(dummies, axis=1)
    X_split = X_split.T.drop_duplicates().T
    X_split = np.array(X_split)

    # CV
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

    print 'without weekday'
    # get dummy variables
    dummies = []
    for col in X_cols[:-1]:
        dummies.append(pd.get_dummies(X_split_no_weekday[col]))

    # Concate data and remove duplicates
    X_split_no_weekday = pd.concat(dummies, axis=1)
    X_split_no_weekday = X_split_no_weekday.T.drop_duplicates().T
    X_split_no_weekday = np.array(X_split_no_weekday)

    # CV
    auc = []
    for train_index, test_index in kf:
        X_train, X_test = X_split_no_weekday[train_index, :], X_split_no_weekday[test_index, :]
        y_train, y_test = y[train_index].ravel(), y[test_index]
        y_test = y_test.astype('float64')

        # train machine learning
        ml.fit(X_train, y_train)

        # predict
        class_pred = ml.predict_proba(X_test)[:, 1]

        # evaluate
        auc.append(roc_auc_score(y_test, class_pred))
    print 'auc is: ', np.mean(auc)

# try to use hour as an int

# VAR_0073.csv
# with weekday
# auc is:  0.633557420088
# without weekday
# auc is:  0.63358292588
# VAR_0075.csv
# with weekday
# auc is:  0.591458805652
# without weekday
# auc is:  0.586035098418
# VAR_0156.csv
# with weekday
# auc is:  0.521305211597
# without weekday
# auc is:  0.521373482187
# VAR_0157.csv
# with weekday
# auc is:  0.505032914601
# without weekday
# auc is:  0.505097542485
# VAR_0158.csv
# with weekday
# auc is:  0.505079087979
# without weekday
# auc is:  0.50491856685
# VAR_0159.csv
# with weekday
# auc is:  0.521372924308
# without weekday
# auc is:  0.521281286924
# VAR_0166.csv
# with weekday
# auc is:  0.543771719974
# without weekday
# auc is:  0.543779412115
# VAR_0167.csv
# with weekday
# auc is:  0.512103132896
# without weekday
# auc is:  0.512103254272
# VAR_0168.csv
# with weekday
# auc is:  0.526349419516
# without weekday
# auc is:  0.526234876782
# VAR_0169.csv
# with weekday
# auc is:  0.543590668204
# without weekday
# auc is:  0.543455878569
# VAR_0176.csv
# with weekday
# auc is:  0.555420096579
# without weekday
# auc is:  0.555405259103
# VAR_0177.csv
# with weekday
# auc is:  0.516328691247
# without weekday
# auc is:  0.51626582353
# VAR_0178.csv
# with weekday
# auc is:  0.529445712606
# without weekday
# auc is:  0.529326858156
# VAR_0179.csv
# with weekday
# auc is:  0.5552237813
# without weekday
# auc is:  0.555079484566
# VAR_0204.csv
# with weekday
# auc is:  0.523556049364
# without weekday
# auc is:  0.523556049364
# VAR_0217.csv
# with weekday
# auc is:  0.532996576706
# without weekday
# auc is:  0.524693494842
