import numpy as np

"""Pre-processing functions used to clean our dataset"""
"""Use function pre_process_data"""

def groupy_by_PRE_jet_num(X,y):
    """Group the data into four sets, depending on whether their PRI_jet_num is {0,1,2,3} 
    Return the four sets and their label"""

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
    return X_0, y_0, X_1, y_1, X_2, y_2, X_3, y_3

def clean_data(tx: np.ndarray, cap:float=0.95) -> np.ndarray:
    """
    cleans the original data by removing the outlier (-999.0) and truncates the variable
    that goes over the cap value (in terms of qunatiles) provided.

    Parameters
    tx: np.ndarray
      input data to clean
    cap: float
      speified qunatile used to cap values beyond this quantile.

    Output
    cleaned: np.ndarray
      cleaned data.
    """
    
    cleaned = np.copy(tx)
    for i, variable in enumerate(cleaned.T):
        # replace outlier with variables mode.
        if len(variable[variable == -999.0]) > 0:
            variable[(variable == -999.0)] = np.round(np.median(variable[variable != -999.0]),2)
        
        # cap outliers. (by given quantile - variable "cap")
        capped_value = np.quantile(variable, cap)
        variable = np.where(variable < capped_value, variable, capped_value)
    return cleaned

def standardize(x):
    """Standardizes the dataset i.e. substract the mean and divide by the standard deviation"""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

  

def pre_process_data(X,y):
    """Puts together all the pre-processing steps and returns the 2 processes datasets"""

    #group by jet_num
    X_0, y_0, X_1, y_1, X_2, y_2, X_3, y_3 = groupy_by_PRE_jet_num(X,y)

    #remove invalid features
    X_0 = clean_data(X_0)
    X_1 = clean_data(X_1)
    X_2 = clean_data(X_2)
    X_3 = clean_data(X_3)

    #standardize the subsets
    X_0,_,_ = standardize(X_0)
    X_1,_,_ = standardize(X_0)
    X_2,_,_ = standardize(X_0)
    X_3,_,_ = standardize(X_0)

    #add a column of 1 to the subsets
    ones = np.ones([len(X_0), 1])
    X_0 = np.append(ones, X_0, axis = 1)
    ones = np.ones([len(X_1), 1])
    X_1 = np.append(ones, X_1, axis = 1)
    ones = np.ones([len(X_2), 1])
    X_2 = np.append(ones, X_2, axis = 1)
    ones = np.ones([len(X_3), 1])
    X_3 = np.append(ones, X_3, axis = 1)

    return X_0, y_0, X_1, y_1, X_2, y_2, X_3, y_3

