import importlib_resources
import numpy as np
import scipy.linalg as linalg

from matplotlib.pyplot import (
    cm,
    figure,
    imshow,
    legend,
    plot,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
    yticks,
)
import xlrd

filename = importlib_resources.files("dtuimldmtools").joinpath("data/Concrete_Data.xls")

# Load xls sheet with data
# There's only a single sheet in the .xls, so we take out that sheet
doc = xlrd.open_workbook(filename).sheet_by_index(0)
print(doc)
# Digits to include in analysis (to include all, n = range(10) )
n = [0, 1]
# Number of principal components for reconstruction
K = 16
# Digits to visualize
nD = range(6)



# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat(filename)["traindata"]
X = traindata[:, 1:]
y = traindata[:, 0]


N, M = X.shape
C = len(n)

classValues = n
classNames = [str(num) for num in n]
classDict = dict(zip(classNames, classValues))


# Select subset of digits classes to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = y == v
    class_mask = class_mask | cmsk
X = X[class_mask, :]
y = y[class_mask]
N = X.shape[0]

# Center the data (subtract mean column values)
Xc = X - np.ones((N, 1)) * X.mean(0)

# PCA by computing SVD of Y
U, S, V = linalg.svd(Xc, full_matrices=False)
# U = mat(U)
V = V.T

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

# Project data onto principal component space
Z = Xc @ V

# Plot variance explained
figure()
plot(rho, "o-")
title("Variance explained by principal components")
xlabel("Principal component")
ylabel("Variance explained value")


# Plot PCA of the data
f = figure()
title("pixel vectors of handwr. digits projected on PCs")
for c in n:
    # select indices belonging to class c:
    class_mask = y == c
    plot(Z[class_mask, 0], Z[class_mask, 1], "o")
legend(classNames)
xlabel("PC1")
ylabel("PC2")


# Visualize the reconstructed data from the first K principal components
# Select randomly D digits.
figure(figsize=(10, 3))
W = Z[:, range(K)] @ V[:, range(K)].T
D = len(nD)
for d in range(D):
    digit_ix = np.random.randint(0, N)
    subplot(2, D, int(d + 1))
    I = np.reshape(X[digit_ix, :], (16, 16))
    imshow(I, cmap=cm.gray_r)
    title("Original")
    subplot(2, D, D + d + 1)
    I = np.reshape(W[digit_ix, :] + X.mean(0), (16, 16))
    imshow(I, cmap=cm.gray_r)
    title("Reconstr.")


# Visualize the pricipal components
figure(figsize=(8, 6))
for k in range(K):
    N1 = int(np.ceil(np.sqrt(K)))
    N2 = int(np.ceil(K / N1))
    subplot(N2, N1, int(k + 1))
    I = np.reshape(V[:, k], (16, 16))
    imshow(I, cmap=cm.hot)
    title("PC{0}".format(k + 1))

# output to screen
show()

print("Ran Exercise 2.2.2")