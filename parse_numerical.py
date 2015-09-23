__author__ = 'YBeer'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.DataFrame.from_csv("train.csv")
print dataset.head()
col_n = dataset.shape[1]

col_names = dataset.columns.values.tolist()
col_types = dataset.dtypes

dataset_temp = dataset[col_names[:100]]
print dataset_temp.shape

dataset_temp = dataset_temp.T.drop_duplicates().T
print dataset_temp.shape

#
# for i, col in enumerate(col_names[40:100]):
#     if col_types[i] == 'int64' or col_types[i] == 'float64':
#         col_data = pd.value_counts(dataset[col])
#         print col_data
