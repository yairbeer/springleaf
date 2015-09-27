__author__ = 'YBeer'

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from datetime import date


classifier = RandomForestClassifier()
classifier_full = GradientBoostingClassifier(loss='deviance', learning_rate=0.2, n_estimators=150, max_depth=3,
                                             max_features=None)

"""
Remove comlumns with only 1 answer
"""
# dataset = pd.DataFrame.from_csv("train.csv")
# print dataset.shape
#
# columns = dataset.columns.values.tolist()
#
# # check column data type
# data_types = dataset.dtypes
#
# bad_columns = 0
# good_columns = []
#
# # check number of values in each column
# for i, col_name in enumerate(columns):
#     # print col_name
#     # print data_types[i]
#     # print dataset[col_name].value_counts().shape[0]
#     if dataset[col_name].value_counts().shape[0] <= 1:
#         bad_columns += 1
#     else:
#         good_columns.append(col_name)
# print 'number of columns with only 1 value: ', bad_columns
#
# # filter bad columns
# dataset = dataset[good_columns]
# dataset.to_csv("train_col_filt.csv")
#
# dataset_test = pd.DataFrame.from_csv("test.csv")
# dataset_test = dataset_test[good_columns[:-1]]
#
# dataset_test.to_csv("test_col_filt.csv")
# print 'written filtered dataframe to file'

"""
Remove duplicate comlumns
"""
# dataset = pd.DataFrame.from_csv("train_col_filt.csv")
#
# col_n = dataset.shape[1]
# interval = 100
#
# col_names = dataset.columns.values.tolist()
# col_types = dataset.dtypes
#
# dataset_splited = []
# for i in range(0, col_n, interval):
#     if col_n > i + interval:
#         dataset_temp = dataset[col_names[i: (i + interval)]]
#         print i, ' before: ', dataset_temp.shape
#
#         dataset_temp = dataset_temp.T.drop_duplicates().T
#         print i, ' after: ', dataset_temp.shape
#         dataset_splited.append(dataset_temp)
#     else:
#         dataset_temp = dataset[col_names[i:]]
#         print i, ' before: ', dataset_temp.shape
#
#         dataset_temp = dataset_temp.T.drop_duplicates().T
#         print i, ' after: ', dataset_temp.shape
#         dataset_splited.append(dataset_temp)
# dataset = pd.concat(dataset_splited, axis=1)
#
# del dataset_splited, dataset_temp
#
# col_names = dataset.columns.values.tolist()
#
# dataset.to_csv("train_col_filt_2.csv")
#
# dataset_test = pd.DataFrame.from_csv("test_col_filt.csv")
# dataset_test = dataset_test[col_names[:-1]]
# dataset_test.to_csv("test_col_filt_2.csv")

"""
change categorical variables to dummy variables, meanwhile ignoring variables with more than 20 values
"""
# get file with only relevant rows
print 'reading train dataset'
dataset = pd.DataFrame.from_csv("train_col_filt_2.csv")

print 'reading test dataset'
dataset_test = pd.DataFrame.from_csv("test_col_filt_2.csv")

good_columns = list(dataset.columns.values)

n = dataset.shape[0]

dictionary = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10,
              'NOV': 11, 'DEC': 12}

# check column data type
data_types = dataset.dtypes
dummies = []
dummies_test = []
print 'starting to convert to dummy variables'
for i in range(len(good_columns) - 1):
    # use getdummies in order to convert categorial to workable numerical table
    col_dif_values = dataset[good_columns[i]].value_counts().shape[0]
    # maximum number of columns viable to create dummies
    print good_columns[i], ' has ', col_dif_values, ' columns'
    if data_types[i] == 'object':
        if col_dif_values <= 100:
            print 'working'
            new_dummy = pd.get_dummies(dataset[good_columns[i]]).astype('float64')
            columns_dummy = new_dummy.columns.values.tolist()
            for j in range(len(columns_dummy)):
                columns_dummy[j] = good_columns[i] + '_' + str(columns_dummy[j])
            new_dummy.columns = columns_dummy

            new_dummy_test = pd.get_dummies(dataset_test[good_columns[i]]).astype('float64')
            columns_dummy_test = new_dummy_test.columns.values.tolist()
            for j in range(len(columns_dummy)):
                columns_dummy_test[j] = good_columns[i] + '_' + str(columns_dummy_test[j])
            new_dummy_test.columns = columns_dummy

            # remove categorical column
            dummies.append(new_dummy)
            dummies_test.append(new_dummy_test)
            # add dummy columns
        # else:
        #     val0 = str(np.array(dataset[good_columns[i]])[0])
        #     print val0
        #     if val0[-2:] == '00' or val0 == 'nan':
        #         dataset[good_columns[i]].to_csv(good_columns[i] + '.csv', header=['date'])
        dataset = dataset.drop(good_columns[i], 1)
        dataset_test = dataset_test.drop(good_columns[i], 1)

# add good date channels
date_col = ['VAR_0073', 'VAR_0075', 'VAR_0176']
for col in date_col:
    X = dataset[col]
    X = np.array(X)

    X_test = dataset_test[col]
    X_test = np.array(X_test)

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

    X_split_test = np.ones((n, 5)) * (-1)
    X_split_test = X_split_test.astype('str')
    for i in range(n):
        if str(X[i, 0]) != 'nan':
            cur_datetime = X_test[i, 0]
            cur_date = date(2000 + int(cur_datetime[5:7]), dictionary[cur_datetime[2:5]], int(cur_datetime[:2]))
            X_split_test[i, 0] = cur_date.year
            X_split_test[i, 1] = cur_date.month
            X_split_test[i, 2] = cur_date.day
            X_split_test[i, 3] = cur_datetime[8:10]
            X_split_test[i, 4] = cur_date.weekday()

    # convert to DF
    X_cols = ['day', 'month', 'year', 'hour', 'weekday']
    X_split = pd.DataFrame(X_split, columns=X_cols)
    X_split_test = pd.DataFrame(X_split_test, columns=X_cols)

    print 'with weekday'
    # get dummy variables
    new_dummy = pd.get_dummies(X_split).astype('float64')
    columns_dummy = new_dummy.columns.values.tolist()
    for j in range(len(columns_dummy)):
        columns_dummy[j] = good_columns[i] + '_' + str(columns_dummy[j])
    new_dummy.columns = columns_dummy

    new_dummy_test = pd.get_dummies(X_split_test).astype('float64')
    columns_dummy_test = new_dummy_test.columns.values.tolist()
    for j in range(len(columns_dummy)):
        columns_dummy_test[j] = good_columns[i] + '_' + str(columns_dummy_test[j])
    new_dummy_test.columns = columns_dummy

    dummies.append(new_dummy)
    dummies_test.append(new_dummy_test)

dataset = pd.concat(dummies + [dataset], axis=1)
dataset_test = pd.concat(dummies_test + [dataset_test], axis=1)

columns_dummy = dataset.columns.values.tolist()
columns_dummy_test = dataset_test.columns.values.tolist()

columns_dummy_and = []
for col in columns_dummy:
    if col in columns_dummy_test:
        columns_dummy_and.append(col)
dataset = dataset[columns_dummy_and + ['target']]
dataset_test = dataset_test[columns_dummy_and]

print 'added new ', len(columns_dummy), ' columns'

print 'finished converting dummies'

dataset.to_csv("train_col_dummy.csv")
dataset_test.to_csv("test_col_dummy.csv")

print 'written dataframe with str to dummy to file'

"""
preprocessing pipe for univariante results
"""
# get file with all numerics
print 'loading dataset with dummies from file'
dataset = pd.DataFrame.from_csv("train_col_dummy.csv")

print 'changing to array'
dataset = np.array(dataset)

X = dataset[:, :-1]
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
univariante evaluation
"""
# CV
cv_n = 4
kf = KFold(dataset.shape[0], n_folds=cv_n, shuffle=True)

print 'start univariante evaluation'
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []
for train_index, test_index in kf:
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index].ravel(), y[test_index].ravel()
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)

uni_results = np.ones((dataset.shape[1], cv_n))
for i in range(X.shape[1]):
    if not i % 50:
        print 'var ', i
    for j in range(cv_n):
        # train machine learning
        classifier.fit(X_train_list[j][:, i].reshape((X_train_list[j].shape[0], 1)), y_train_list[j])

        # predict
        class_pred = classifier.predict_proba(X_test_list[j][:, i].reshape((X_test_list[j].shape[0], 1)))[:, 1]
        # evaluate
        uni_results[i, j] = roc_auc_score(y_test_list[j], class_pred)

print uni_results
print np.mean(uni_results, axis=1)
uni_results = np.mean(uni_results, axis=1)

uni_results = pd.Series(uni_results)
print uni_results.value_counts()
uni_results.to_csv("univar_AUC.csv")

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
print regression_matrix_indices

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

PCA
print 'PCA results'
pca_decomp = PCA(n_components=100)
X = pca_decomp.fit_transform(X)
print X.shape

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
print 'fitting full data'
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
# X_test = PCA.transform(X_test)

# predict
class_pred = classifier_full.predict_proba(X_test)[:, 1]

submission_file = pd.DataFrame.from_csv("sample_submission.csv")
submission_file['target'] = class_pred
submission_file.to_csv("rf_univar_" + str(uni_thresh) + ".csv")
