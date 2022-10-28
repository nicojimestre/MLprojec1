import numpy as np

"""Pre-processing functions used to clean our dataset"""
"""Use function pre_process_data"""

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

    #remove invalid features
    X = clean_data(X)
    y = clean_data(y)
    #standardize the subsets
    X,_,_ = standardize(X)
    y,_,_ = standardize(y)

    return X, y

