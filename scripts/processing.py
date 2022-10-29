import numpy as np
from numpy import NaN
from feature_expansion import *

"""Pre-processing functions used to clean our dataset"""
"""Use function pre_process_data"""

def group_by_PRE_jet_num(X,y):
    """Group the data into four sets, depending on whether their PRI_jet_num is {0,1,2,3} 
    Return the four sets and their label"""
    #for i in range(0, len(X)):
        
    #create masks to extract the subsets
    mask_0 = X[:,22] == 0
    mask_1 = X[:,22] == 1
    mask_2 = X[:,22] == 2
    mask_3 = X[:,22] == 3
    
    #extract the elements from each subset
    X_0 = X[mask_0,:]
    X_1 = X[mask_1,:]
    X_2 = X[mask_2,:]
    X_3 = X[mask_3,:]

    #extract the corresponding labels
    if(len(y)>0):
        y_0 = np.asarray(y[mask_0])
        y_1 = np.asarray(y[mask_1])
        y_2 = np.asarray(y[mask_2])
        y_3 = np.asarray(y[mask_3])
    #print(np.isnan(X_0))
    return X_0, y_0, X_1, y_1, np.concatenate((X_2, X_3), axis = 0), np.concatenate((y_2, y_3), axis=0)

def clean_data(tx: np.ndarray, features, cap:float=0.95) -> np.ndarray:
    """
    cleans the original data by removing the outlier (-999.0) and truncates the variable
    that goes over the cap value (in terms of qunatiles) provided.

    Parameters
    tx: np.ndarraye
      input data to clean
    cap: float
      speified qunatile used to cap values beyond this quantile.

    Output
    cleaned: np.ndarray
      cleaned data.
    """
    quartiles = np.ndarray([len(features), 2])
    cleaned = np.copy(tx)
    for i, variable in enumerate(cleaned.T):
        quartiles[i] = np.quantile(variable, [0.75]) 
    upper_bound = [x[1]*1.5 for x in quartiles]
    to_modify = [0,1,2,3,5,8,9,10,13,16,19,21,23,26,20]
    for i in to_modify:
        u_capped_value = upper_bound[i]
        variable = np.where(variable < u_capped_value, variable, u_capped_value)
    cleaned = np.delete(cleaned, features.index('PRI_jet_num'), axis = 1)
    j = 0
    for i, variable in enumerate(cleaned.T):
        # replace outlier with variables mode.
        #If all columns have values of -999, we delete the feature
        if len(variable[variable == -999.0]) > 0:
            if len(variable[variable != -999.0]) == 0:
                cleaned = np.delete(cleaned, j, axis = 1)
                j -= 1
            else:
                variable[(variable == -999.0)] = np.round(np.median(variable[variable != -999.0]),2)
        elif np.std(variable, axis=0) == 0:
            cleaned = np.delete(cleaned, j, axis = 1)    
            j -= 1
        j += 1
    
    return cleaned

def standardize(x):
    """Standardizes the dataset i.e. substract the mean and divide by the standard deviation"""
    
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x
  

def pre_process_data(X,y, features):
    """Puts together all the pre-processing steps and returns the 2 processes datasets"""
    #group by jet_num
    #print(NaN in test[27])
    X_0, y_0, X_1, y_1, X_23, y_23 = groupy_by_PRE_jet_num(X,y)
    #print(np.isnan(clean_data(X_1, features)))
    #remove invalid features
    X_0 = clean_data(X_0, features)
    X_1 = clean_data(X_1, features)
    X_23 = clean_data(X_23, features)
    #print X_0[27])
    
    #print(-999.00 in X_0)
    X_0_extended = build_new_x(X_0, features)
    X_1_extended = build_new_x(X_1, features)
    X_23_extended = build_new_x(X_23, features)
    print(len(X_0[0]), len(X_0))
    

    #standardize the subsets
    X_0_extended[0:len(X_0), 0:len(X_0[0])] = standardize(X_0)
    X_1_extended[0:len(X_1), 0:len(X_1[0])] = standardize(X_1)
    X_23_extended[0:len(X_23), 0:len(X_23[0])] = standardize(X_23)

    #add a column of 1 to the subsets
    ones = np.ones([len(X_0), 1])
    X_0_extended = np.append(X_0_extended, ones, axis = 1)
    ones = np.ones([len(X_1), 1])
    X_1_extended = np.append(X_1_extended, ones, axis = 1)
    ones = np.ones([len(X_23), 1])
    X_23_extended = np.append(X_23_extended, ones, axis = 1)
    

    return X_0_extended, y_0, X_1_extended, y_1, X_23_extended, y_23

