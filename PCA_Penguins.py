
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
file_path = os.path.join(path, "penguins.xls")

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

# Check the number of columns in the DataFrame
num_columns = len(data.columns)

# Now you can proceed with the rest of your code
# Define features and target variable
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
target = ['species']

# Extract features (X) and target variable (y)
X = data.loc[:, features].values
y = data.loc[:,target].values
# Standardize features

x = StandardScaler().fit_transform(X)

# PCA by computing SVD of x
U, S, V = linalg.svd(x, full_matrices=False)

# U = mat(U)
V = V.T

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()


# Project data onto principal component space
Z = x @ V

# Plot variance explained
threshold = 0.65
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

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

classNames=list(["Adelie","Gentoo","Chinstrap"])
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
ax.legend(classNames)
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

plt.show()

# Next step is to plot the variance explained by each principal component.
#https://convertio.co/download/f60700f073232ef69c69648a2f4494378e1a7d/

#PCA component coefficients in histogram
# Barplot
aa = np.arange(len(features))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(layout='constrained')
for vector in U:
    offset = width * multiplier
    rects = ax.bar(aa + offset, vector, width)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Component coefficients')
ax.set_title('PCA components coefficients')
ax.set_xticks(aa + width, features,rotation=45)
ax.grid(axis='y')
ax.set_yticks(np.arange(-0.5, 0.8, 0.25))

plt.show()


### DATA VISUALIZATION --------------------------------------------------------
# Histogram
plt.figure(figsize=(8, 4))
M = len(features)
for i in range(M):
    plt.subplot(1, 4, i+1)
    plt.hist(X[:, i], color=(0.2, 0.8 - i * 0.2, 0.4))
    plt.xlabel(features[i])
    plt.ylim([0,80])
plt.show()

# matrix of scatter plots
classNames=list(["Adelie","Gentoo","Chinstrap"])
C=len(classNames)
plt.figure(figsize=(12, 10))
M = len(features)
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = y == c
            class_mask = class_mask[:,0]
            plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), ".")
            if m1 == M - 1:
                plt.xlabel(features[m2])
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(features[m1])
            else:
                plt.yticks([])
            # ylim(0,X.max()*1.1)
            # xlim(0,X.max()*1.1)
plt.legend(classNames)

plt.show()

#BoxPlot
plt.figure(figsize=(14, 7))
for c in range(C):
    plt.subplot(1, C, c+1)
    class_mask = y == c  # binary mask to extract elements of class c
    class_mask = class_mask[:,0]
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c

    plt.boxplot(x[class_mask])
    # title('Class: {0}'.format(classNames[c]))
    plt.title("Class: " + classNames[c])
    plt.xticks(
        range(1,M+1), [a for a in features], rotation=45
    )
    y_up = x.max() + (x.max() - x.min()) * 0.1
    y_down = x.min() - (x.max() - x.min()) * 0.1
    plt.ylim(y_down, y_up)

plt.show()

