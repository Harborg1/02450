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
import random


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

# print(Y_r)

# print(X_r)

print(X_r[:, 0])
print(X_r[:, 1])
print(X_r[:, 2])
print(X_r[:, 3])
print(X_r[:, 4])
print(X_r[:, 5])
print(Y_r)

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
results = np.empty((K, 6))

k = 0

n_hidden_units = random.randint(1, 5)

# Define the model with dynamically set number of hidden units
model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, n_hidden_units),   # Input layer to hidden layer
    torch.nn.BatchNorm1d(n_hidden_units), # Batch normalization for first hidden layer
    torch.nn.Linear(n_hidden_units, 1)    # Hidden layer to output layer
)
loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

# Parameters for neural network classifier

n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000
cvf_outer =10
cvf_inner = 10

#--------------------------------------------------------------------------------------------

print("Training model of type:\n\n{}\n".format(str(model())))
errors = []  # make a list for storing generalizaition error in each loop

CV_outer = model_selection.KFold(cvf_outer, shuffle=True)
w = np.empty((M, 10, len(lambdas)))
train_error = np.empty((10, len(lambdas)))
test_error = np.empty((10, len(lambdas)))


for f_outer,(train_index, test_index) in enumerate(CV.split(X_r, Y_r)):
    # extract training and test set for current CV fold
    X_train = X_r[train_index]
    y_train = Y_r[train_index]
    X_test = X_r[test_index]
    y_test = Y_r[test_index]
    CV_inner = model_selection.KFold(cvf_outer, shuffle=True)
    #ANN------------------------------------------------------------------------------------------
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

    mse = np.mean(se.detach().numpy())

    print("MSE", mse)

    print("Final loss", final_loss)

#errors.append(mse)  # store error rate for current CV fold

    print("Final loss", final_loss)


    for train_index_inner, test_index_inner in (CV_inner.split(X_train, y_train)):
            X_train_inner, X_val = X_train[train_index_inner], X_train[test_index_inner]
            y_train_inner, y_val = y_train[train_index_inner], y_train[test_index_inner]
            
            mu = np.mean(X_train_inner[:, 1:], axis=0)
            sigma = np.std(X_train_inner[:, 1:], axis=0)
            
        
            X_train_inner[:, 1:] = (X_train_inner[:, 1:] - mu) / sigma
            X_val[:, 1:] = (X_val[:, 1:] - mu) / sigma
            X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
            Xty = X_train_inner.T @ y_train_inner
            XtX = X_train_inner.T @ X_train_inner
            for l in range(len(lambdas)):
                lambdaI = lambdas[l] * np.eye(M)
                lambdaI[0, 0] = 0
                w[:, f_outer, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
                train_error[f_outer, l] = np.power(y_train_inner - X_train_inner @ w[:, f_outer, l].T, 2).mean(axis=0)
                test_error[f_outer, l] = np.power(y_val - X_val @ w[:, f_outer, l].T, 2).mean(axis=0)

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))
    
   
# #errors.append(mse)  # store error rate for current CV fold

# #---------------------------------------------------------------------------------------------

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    if k==2:
        k=k-1
        
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    Error_test_rlr[k] = (
        np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )

    print("Error test:", Error_test_rlr[k])

    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )
    
results[k, 0]  = k
results[k, 1]  = n_hidden_units
print("Added", n_hidden_units, "to results" )
results[k, 2]  = mse
print("Added", mse, "to results")
results[k, 3]  =opt_lambda
print("Added", opt_lambda,"to results")
results[k, 4]  =Error_test_rlr[k]
results[k, 5]  =Error_test_nofeatures[k]
   
    # Display the results for the last cross-validation fold
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
        k+=1
        

    # To inspect the used indices, use these print statements
    # print('Cross validation fold {0}/{1}:'.format(k+1,K))
    # print('Train indices: {0}'.format(train_index))
    # print('Test indices: {0}\n'.format(test_index))

show()


print("First row of results:")
print(results[0,1])
print(results[0,2])
print(results[0,3])
print(results[0,4])
print(results[1,1])
print(results[1,2])
print(results[1,3])
print(results[1,4])
# slopes = []

# for i in range(len(lambdas) - 1):
#     # Calculate slope between consecutive points
#     slope = (test_err_vs_lambda[i + 1] - test_err_vs_lambda[i]) / (lambdas[i + 1] - lambdas[i])
#     slopes.append(slope)

# # Print the calculated slopes
# print("Slopes between consecutive points:", slopes)

# #ANN---------------------------------------------------------------------------------------------
# print(
#     "\nEstimated generalization error, RMSE: {0}".format(
#         round(np.sqrt(np.mean(errors)), 4)
#     )
# )

#-------------------------------------------------------------------------------------------------
