# import importlib_resources
# import numpy as np
# import scipy.linalg as linalg
# import pandas as pd
# from matplotlib.pyplot import (
#     cm,
#     figure,
#     imshow,
#     legend,
#     plot,
#     show,
#     subplot,
#     title,
#     xlabel,
#     ylabel,
#     yticks,
# )
# import xlrd

# filename = importlib_resources.files("dtuimldmtools").joinpath("data/penguins.xls")

# # Load xls sheet with data
# # There's only a single sheet in the .xls, so we take out that sheet
# doc = xlrd.open_workbook(filename).sheet_by_index(0)
# print(doc)
# # Digits to include in analysis (to include all, n = range(10) )
# n = [0, 1]
# # Number of principal components for reconstruction
# K = 16
# # Digits to visualize
# nD = range(6)

# traindata = pd.read_excel(filename)
# print(traindata)
# X= traindata[:, 1:]
# y = traindata[:, 0]  # Assuming labels are in the first column

# print(X)
# print(y)


import importlib_resources
import numpy as np
import scipy.linalg as linalg
import pandas as pd
from LoadFile.py import X,y

## Standarize data
# Subtract the mean from the data
X = X - np.ones((N, 1)) * X.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
X = X * (1 / np.std(X, 0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions of X
