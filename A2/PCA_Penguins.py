
# SOURCES:
# https://builtin.com/machine-learning/pca-in-python
# exercise 2.2.2 

from matplotlib import figure
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np
import os

path=os.getcwd()
print(path)
file_path = os.path.join(path, "penguinsNew.xls")

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

Y = data.iloc[:, 5] 

X = data.iloc[:,[0,2,3,4]]

# Standardize the numerical data
x_numeric = StandardScaler().fit_transform(data.iloc[:, [2,3,4]])

# The categorical data (species)
x_categorical = data.iloc[:,[0]]

x_processed = np.concatenate((x_categorical, x_numeric), axis=1) #Concatenate numeric and categorical data

data = np.concatenate((x_processed, np.expand_dims(Y, axis=1)), axis=1)

Y_r  = data[:, 4] # Get the body mass (target)
 
X_r = data[:,[0,1,2,3]] # Get the features (species, bill_length, bill_depth, flipper_length)

species = np.array(X_r[:, 0], dtype=int).T

K = species.max() + 1

species_encoding = np.zeros((species.size, K))

species_encoding[np.arange(species.size), species] = 1

print(species_encoding)

X_r = np.concatenate((X_r[:, :1], species_encoding, X_r[:, 1:]), axis=1)
X_r[:, 0] = X_r[:, 1] # Make the first column equal the second column

X_r = np.delete(X_r, 1, axis=1) # Delete the second column

#print(X_r)
print(X_r[:, 0])
print(X_r[:, 1])
print(X_r[:, 2])
print(X_r[:, 3])
print(X_r[:, 4])
print(X_r[:, 5])


