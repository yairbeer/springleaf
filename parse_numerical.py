__author__ = 'YBeer'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.DataFrame.from_csv("train.csv")
print dataset.head()

col_n = dataset.shape[1]
interval = 100

col_names = dataset.columns.values.tolist()
col_types = dataset.dtypes

dataset_splited = []
for i in range(0, col_n, interval):
    if col_n > i + interval:
        dataset_temp = dataset[col_names[i: (i + interval)]]
        print i, ' before: ', dataset_temp.shape

        dataset_temp = dataset_temp.T.drop_duplicates().T
        print i, ' after: ', dataset_temp.shape
        dataset_splited.append(dataset_temp)
    else:
        dataset_temp = dataset[col_names[i:]]
        print i, ' before: ', dataset_temp.shape

        dataset_temp = dataset_temp.T.drop_duplicates().T
        print i, ' after: ', dataset_temp.shape
        dataset_splited.append(dataset_temp)
dataset = pd.concat(dataset_splited, axis=1)

del dataset_splited, dataset_temp
print dataset.head()

#
# for i, col in enumerate(col_names[40:100]):
#     if col_types[i] == 'int64' or col_types[i] == 'float64':
#         col_data = pd.value_counts(dataset[col])
#         print col_data
