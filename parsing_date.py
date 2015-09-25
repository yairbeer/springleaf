import pandas as pd

__author__ = 'yaia'

dataset = pd.DataFrame.from_csv("VAR_0073.csv")

columns = dataset.columns.values.tolist()

target = dataset[columns[-1]]

target.to_csv("target.csv", header=['target'])
