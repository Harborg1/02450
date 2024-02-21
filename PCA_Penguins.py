
# SOURCES:
# https://builtin.com/machine-learning/pca-in-python
# exercise 2.2 (a)

from matplotlib import figure
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np
import os

path=os.getcwd()
file_path = os.path.join(path, "penguins.xls")

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

# Check the number of columns in the DataFrame
num_columns = len(data.columns)

# Now you can proceed with the rest of your code
# Define features and target variable
features = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']
target = ['species']

# Extract features (X) and target variable (y)
X = data.loc[:, features].values
y = data.loc[:,target].values
# Standardize features

x = StandardScaler().fit_transform(X)

# PCA by computing SVD of Y
U, S, V = linalg.svd(x, full_matrices=False)

# U = mat(U)
V = V.T

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

# Project data onto principal component space
Z = x @ V

# Plot variance explained
plt.figure()
plt.plot(rho, "o-")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained value")


pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

U = pca.components_

explained_variance = pca.explained_variance_ratio_

print("Variance explained by the first principal component:", explained_variance[0])
print("Variance explained by the second principal component:", explained_variance[1])


#print(principalComponents)


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, data[['species']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PCA1', fontsize = 15)
ax.set_ylabel('PCA2', fontsize = 15)
ax.set_title('PCA analysis', fontsize = 20)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):   
    indicesToKeep = finalDf['species'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

plt.show()

# Next step is to plot the variance explained by each principal component.
#https://convertio.co/download/f60700f073232ef69c69648a2f4494378e1a7d/



