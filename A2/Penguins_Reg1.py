# SOURCES:
# https://builtin.com/machine-learning/pca-in-python
# exercise 2.2.2 

from matplotlib import figure
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np
import os
from sklearn import linear_model, model_selection

from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
    xlim,
    ylim
)

path=os.getcwd()
print(path)
file_path = os.path.join(path, "penguinsNew.xls")

# Read the Excel file into a DataFrame
data = pd.read_excel(file_path)

y = data.iloc[:, 5] # The body weight
#print(Y)
X = data.iloc[:,[0,2,3,4]]

# The categorical data (species)

x_categorical = data.iloc[:,[0]]

data = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)

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

X_r = np.concatenate((np.ones((X_r.shape[0], 1)), X_r), 1)

print(Y_r)

print(X_r)

print(X_r[:, 0])
# print(X_r[:, 1])
# print(X_r[:, 2])
# print(X_r[:, 3])
# print(X_r[:, 4])
# print(X_r[:, 5])
# print(Y_r)

def rlr_validate(X, y, lambdas, cvf=10):
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M, cvf, len(lambdas)))
    train_error = np.empty((cvf, len(lambdas)))
    test_error = np.empty((cvf, len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train - X_train @ w[:, f, l].T, 2).mean(
                axis=0
            )
            test_error[f, l] = np.power(y_test - X_test @ w[:, f, l].T, 2).mean(axis=0)

        f = f + 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

    return (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    )

N, M = X_r.shape
M = M + 1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
# Values of lambda
lambdas = np.power(10.0, range(-2, 2))
# Initialize variables
# T = len(lambdas)
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0

for train_index, test_index in CV.split(X_r, Y_r):
    # extract training and test set for current CV fold
    X_train = X_r[train_index]
    y_train = Y_r[train_index]
    X_test = X_r[test_index]
    y_test = Y_r[test_index]
    internal_cross_validation = 10
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    

    # Display the results for the last cross-validation fold
    if k == K - 1:
        
        plt.title("Optimal lambda: 1e{0}".format(opt_lambda))
        # loglog(
        #     lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
        # )
        plt.plot(
            lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
        )
        plt.xlabel("Regularization factor")
        plt.ylabel("Squared error (crossvalidation)")
        plt.legend(["Train error", "Validation error"])
        plt.grid()
        
    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

    k += 1
show()

slopes = []

for i in range(len(lambdas) - 1):
    # Calculate slope between consecutive points
    slope = (test_err_vs_lambda[i + 1] - test_err_vs_lambda[i]) / (lambdas[i + 1] - lambdas[i])
    slopes.append(slope)

# Print the calculated slopes
print("Slopes between consecutive points:", slopes)