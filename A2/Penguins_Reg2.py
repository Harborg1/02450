
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
import torch
from dtuimldmtools import draw_neural_net, train_neural_net
from scipy import stats


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

import sys

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

X_r = data[:,[1,2,3]] # Get the features (species, bill_length, bill_depth, flipper_length)
# Normalize data
X_r = stats.zscore(X_r)
X_r = np.concatenate((data[:,0:1],X_r),1)


species = np.array(X_r[:, 0], dtype=int).T

K = species.max() + 1

species_encoding = np.zeros((species.size, K))

species_encoding[np.arange(species.size), species] = 1

print(species_encoding)

X_r = np.concatenate((X_r[:, :1], species_encoding, X_r[:, 1:]), axis=1)
X_r[:, 0] = X_r[:, 1] # Make the first column equal the second column

X_r = np.delete(X_r, 1, axis=1) # Delete the second column

X_r = np.concatenate((np.ones((X_r.shape[0], 1)), X_r), 1)

N, M = X_r.shape


# print(Y_r)

# print(X_r)

#print(X_r[:, 0])
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
        X_train_inner = X[train_index]
        y_train_inner = y[train_index]
        X_test_inner = X[test_index]
        y_test_inner = y[test_index]

        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train_inner[:, 1:], 0)
        sigma = np.std(X_train_inner[:, 1:], 0)

        X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu) / sigma
        X_test_inner[:, 1:] = (X_test_inner[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train_inner.T @ y_train_inner
        XtX = X_train_inner.T @ X_train_inner
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train_inner - X_train_inner @ w[:, f, l].T, 2).mean(
                axis=0
            )
            test_error[f, l] = np.power(y_test_inner - X_test_inner @ w[:, f, l].T, 2).mean(axis=0)

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
        # X_train_inner,
        # y_train_inner
    )

def ANN(X, y, n_replicates, max_iter):
    errors_inner = []  # make a list for storing generalizaition error in each loop
    #print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))
    
    for train_index, test_index in CV.split(X, y):
        # Extract training and test set for current CV fold, convert to tensors
        X_train_ANN_inner = torch.Tensor(X[train_index, :])
        y_train_ANN_inner = torch.Tensor(y[train_index])
        X_test_ANN_inner = torch.Tensor(X[test_index,:])
        y_test_ANN_inner = torch.Tensor(y[test_index])
    
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(
            model,
            loss_fn,
            X=X_train_ANN_inner,
            y=y_train_ANN_inner,
            n_replicates=n_replicates,
            max_iter=max_iter,
            )

        print("\n\tBest loss: {}\n".format(final_loss))

        # Determine estimated class labels for test set
        y_test_est_inner = net(X_test_ANN_inner)
        y_test_est_inner = y_test_est_inner[:,0]

        # Determine errors and errors
        se_inner = (y_test_est_inner.float() - y_test_ANN_inner.float()) ** 2  # squared error
        mse_inner = (sum(se_inner).type(torch.float) / len(y_test_ANN_inner)).data.numpy()  # mean
        errors_inner.append(mse_inner)  # store error rate for current CV fold
    return(
        errors_inner      
        )
    #---------------------------------------------------------------------------------------------
        



## Crossvalidation
# Create crossvalidation partition for evaluation
K = 2
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
results = np.empty((K, 4))

k = 0

# Used in ANN -------------------------------------------------------------------------------
# Define the model
model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
    torch.nn.Tanh(),  # 1st transfer function,
    torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
    # no final tranfer function, i.e. "linear output"
)
loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

# Parameters for neural network classifier
n_hidden_units = 2  # number of hidden units
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000
#--------------------------------------------------------------------------------------------

print("Training model of type:\n\n{}\n".format(str(model())))
errors = []  # make a list for storing generalizaition error in each loop


for train_index, test_index in CV.split(X_r, Y_r):
    # extract training and test set for current CV fold
    X_train_outer = X_r[train_index]
    y_train_outer = Y_r[train_index]
    X_test_outer = X_r[test_index]
    y_test_outer = Y_r[test_index]
    internal_cross_validation = 10
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
        # X_train_inner,
        # y_train_inner,
    ) = rlr_validate(X_train_outer, y_train_outer, lambdas, internal_cross_validation)
    
    #ANN------------------------------------------------------------------------------------------
    errors_inner = ANN(
        X=X_train_outer,
        y=y_train_outer,
        n_replicates=n_replicates,
        max_iter=max_iter
    )
# stop code here - move up to break code where you like
    sys.exit()    
    
    
    
    
    
    print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))
    # Extract training and test set for current CV fold, convert to tensors
    X_train_ANN = torch.Tensor(X_r[train_index, :])
    y_train_ANN = torch.Tensor(Y_r[train_index])
    X_test_ANN = torch.Tensor(X_r[test_index, :])
    y_test_ANN = torch.Tensor(Y_r[test_index])
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train_ANN,
        y=y_train_ANN,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    print("\n\tBest loss: {}\n".format(final_loss))

    # Determine estimated class labels for test set
    y_test_est = net(X_test_ANN)

    # Determine errors and errors
    se = (y_test_est.float() - y_test_ANN.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_test_ANN)).data.numpy()  # mean
    errors.append(mse)  # store error rate for current CV fold
    
    #---------------------------------------------------------------------------------------------
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    
    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )
    
    
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )
    
    results[k, 0] = k
    results[k, 1] = opt_lambda
    results[k, 2]  = Error_test_rlr[k]
    results[k, 3] = Error_test_nofeatures[k]
    
    # Display the results for the last cross-validation fold
    if k == K - 1:
        
        plt.title("Optimal lambda: 1e{0}".format(opt_lambda))
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


print("the results is", results)

slopes = []

for i in range(len(lambdas) - 1):
    # Calculate slope between consecutive points
    slope = (test_err_vs_lambda[i + 1] - test_err_vs_lambda[i]) / (lambdas[i + 1] - lambdas[i])
    slopes.append(slope)

# Print the calculated slopes
print("Slopes between consecutive points:", slopes)

#ANN---------------------------------------------------------------------------------------------
print(
    "\nEstimated generalization error, RMSE: {0}".format(
        round(np.sqrt(np.mean(errors)), 4)
    )
)

#-------------------------------------------------------------------------------------------------
