"""Cross Validation methods to optimize hyperparameters"""

import numpy as np

from scripts.implementations import compute_mse_loss, ridge_regression

def split_validation(y, x, k_indices, k):
    """returns the training set and the test set with the k_fold value of the cross validation"""

    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)

    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    return x_tr, y_tr, x_te,y_te

def poly_expansion(x,degree):
    """Polynomial feature expansion: Takes a matrix X and returns a new matrix (1,X,X^2,...,X^degree)"""
    poly = np.ones((len(x),1))
    degree = int(degree)
    for deg in range(1,degree+1):
        poly = np.c_[poly,np.power(x,deg)]
    return poly

def log_transform(x, features):
    """Apply a log transform to all features of x that have been specified in the arguments. Returns a new matrix and doesn't modify the given x"""
    #For all the features that have negative values, we take their opposite in order to give only positive values to the log-function
    new = np.copy(x)
    for f in features:
        column = x[:,f]
        column = np.abs(column)
        column = np.log(1+column)
        
        new[:,f] = column
        
    return new   

def cross_validation_ridge(k_fold,tX,y,seed=125):
    """grid-search cross validation to find the best hyperparameters (degree and lambda) for ridge regression with a polynomial     expansion """
    degrees  = np.arange(10)
    lambdas = np.logspace(-15,0,16)
    degrees_plus_lambdas = np.transpose([np.tile(degrees,len(lambdas)),np.repeat(lambdas,len(degrees))])
    
    #get indices of k-subgroups of our features and labels datasets
    num_row = y.shape[0]
    interval = int(num_row / 10)
    indices = np.random.permutation(num_row)
    k_indices = np.array([indices[k * interval: (k + 1) * interval] for k in range(10)])
    
    rmse_tr = []
    rmse_te = []

    best = 2**30
    best_degree = -1
    best_lambda = -1
    
    #iterate over all possible degree, lambda pairs and keep the best one
    for pair in degrees_plus_lambdas:
        degree = pair[0]
        lambda_ = pair[1]

        rmse_tr_temp = []
        rmse_te_temp = []
    
        #go over all possible folds for each hyperparameter combination and compute the mean error  
        for k in range(k_fold):
            #create training and validation set
            x_tr,y_tr,x_te,y_te = split_validation(y,tX,k_indices,k)
            
            #polynomial expansion with the degree hyperparameter
            x_tr = poly_expansion(x_tr,degree)
            x_te = poly_expansion(x_te,degree)
            
            #log transformation on the expanded matrix
            x_tr_log = log_transform(x_tr, np.arange(x_tr.shape[1]))
            x_te_log = log_transform(x_te, np.arange(x_te.shape[1]))
            
            x_tr = np.c_[x_tr,x_tr_log]
            x_te = np.c_[x_te,x_te_log]

            weights, _ = ridge_regression(y_tr, x_tr, lambda_)

            loss_tr = np.sqrt(2*compute_mse_loss(y_tr,x_tr,weights))
            loss_te = np.sqrt(2*compute_mse_loss(y_te,x_te,weights))

            rmse_tr_temp.append(loss_tr)
            rmse_te_temp.append(loss_te)

        rmse_tr_mean = np.mean(rmse_tr_temp)
        rmse_te_mean = np.mean(rmse_te_temp)
        
        #if we get a new smallest rmse, keep track of the parameters that gave us the optimal solution
        if(rmse_te_mean < best):
            best = rmse_te_mean
            best_degree = degree
            best_lambda = lambda_
        
        print("Degree:",degree," Lambda:",lambda_," RMSE Training:",rmse_tr_mean, " RMSE Test:", rmse_te_mean)
        rmse_tr.append([degree,lambda_,rmse_tr_mean])
        rmse_te.append([degree,lambda_, rmse_te_mean])
    
    return [best,best_degree,best_lambda,np.asarray(rmse_tr),np.asarray(rmse_te)]

