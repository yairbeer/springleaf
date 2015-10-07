import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
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
# rows = random.sample(dataset.index, 30000)
# dataset = dataset.ix[rows]

print 'changing to array'
dataset = np.array(dataset)

item_list = range(100, 210, 20)
for item in item_list:

    print item
    classifier = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.6, n_iter=item)

    uni_thresh = 0.3
    print 'threshold is ', uni_thresh
    regression_matrix_indices = []
    for i in range(len(uni_results) - 1):
        if uni_results['AUC'][i] > uni_thresh:
            regression_matrix_indices.append(i)
    print len(regression_matrix_indices)

    X = dataset[:, regression_matrix_indices]
    y = dataset[:, -1]

    # impotate
    print 'impotating'
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    X = imp.transform(X)

    # standardizing results
    print 'standardizing results'
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    # # PCA
    # print 'PCA results'
    # pca_decomp = PCA(n_components=10)
    # X_pca = pca_decomp.fit_transform(X)
    #
    # X = np.hstack((X, X_pca))
    print X.shape

    """
    full model CV
    """
    # CV
    cv_n = 4
    kf = KFold(dataset.shape[0], n_folds=cv_n, shuffle=True)
    print 'start full model evaluation'

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
    print item, ' auc is: ', np.mean(auc)
