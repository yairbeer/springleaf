__author__ = 'YBeer'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.DataFrame.from_csv("train.csv")
print dataset.head()

dataset = dataset.T
dataset = dataset.drop_duplicates()
dataset = dataset.T
print dataset.head()

# col_names = dataset.columns.values.tolist()
#
# col_types = dataset.dtypes
#
# for i, col in enumerate(col_names[40:100]):
#     if col_types[i] == 'int64' or col_types[i] == 'float64':
#         col_data = pd.value_counts(dataset[col])
#         print col_data
