"""Cross Validation methods to optimize hyperparameters"""

#from scripts.feature_expansion import build_log_transformation
from cmath import isnan
from numpy import NaN
from processing import standardize
import numpy as np
from itertools import product

from metrics import *
from implementations import *
from feature_expansion import *

class HyperParameterTuner:
    def __init__(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        model_name: str,
        num_folds: int,
        num_seed: int=12,
        max_iter: int=1000
    ):

        available_models = {
        'least_squares': least_squares,
        'ridge': ridge_regression,
        'mse_gd': mean_squared_error_gd,
        'mse_sgd': mean_squared_error_sgd,
        'logistic': logistic_regression,
        'reg_logistic': reg_logistic_regression
        }

        self.x = x
        self.y = y
        self.x_stdr = np.c_[x.T[0], standardize(x[0: len(x), 1: len(x[0])])]
        self.model = available_models[model_name]
        self.model_name = model_name
        self.num_folds = num_folds
        self.num_seed = num_seed
        self.max_iter = max_iter

        # build k_indices
        
        self.build_k_indices()

        # get model params given model specs.
        model_parameters = {
            'least_squares': {}, 
            'ridge'        : {'lambda_': None},
            'mse_gd'       : {'initial_w': np.ones((len(y),1)), 'max_iters': self.max_iter, 'gamma': None},
            'mse_sgd'      : {'initial_w': np.ones((len(y),1)), 'max_iters': self.max_iter, 'gamma': None},
            'logistic'     : {'initial_w': np.ones((len(y),1)), 'max_iters': self.max_iter, 'gamma': None},
            'reg_logistic' : {'initial_w': np.ones((len(y),1)), 'max_iters': self.max_iter, 'gamma': None, 'lambda_': None},
        }
        self.hyp_params = model_parameters[self.model_name]
        self.model = available_models[self.model_name]

        self.degrees = np.arange(1,4)
        self.degree = 0
        

    def tune_(self) -> Tuple[list, float]:
        """
        hyperparameter tuning done by grid search.
        best parameters are found by finding the maximum f1 scores.
        """
        lambdas = np.logspace(-15,0,15)
        gammas  = np.linspace(0,1,100)    
        pairs = np.transpose([np.tile(self.degrees, len(lambdas)),np.repeat(lambdas, len(self.degrees))]) 

        f1_scores = []
        params = self.hyp_params
        # cross validation
        if self.model_name == 'least_squares':
            results = np.array([[k, self.cross_validation_per_k(k, self.hyp_params)[-1]] for k in range(self.num_folds)])
            return results[np.argmax(results[:,-1])]   
        
        elif self.model_name == 'reg_logistic':
            lambda_and_gammas = product(gammas, lambdas)
            for (gamma, lambda_) in lambda_and_gammas:
                params['gamma'], params['lambda_'] = gamma, lambda_
                results = np.concatenate([[self.cross_validation_per_k(k, params)] for k in range(self.num_folds)], axis=1)
                f1_scores.append(np.mean(results, axis=0))
        
            optimum_idx = np.argmax(f1_scores)
            best_params, best_f1 = lambda_and_gammas[optimum_idx], f1_scores[optimum_idx]
            return best_params, best_f1     

        elif self.model_name == 'ridge':
            for pair in pairs:
                lambda_ = pair[1]
                self.degree = int(pair[0])
                params['lambda_'] =lambda_
                results = np.concatenate([[self.cross_validation_per_k(k, params)] for k in range(self.num_folds)], axis=1)
                f1_scores.append(np.mean(results, axis=0)[-1])

                print("we are in ", lambda_)
            
            f1_scores = f1_scores[np.logical_not(np.isnan(f1_scores))]
            optimum_idx = np.argmax(f1_scores)
            best_params, best_f1 = lambdas[optimum_idx], f1_scores[optimum_idx]
            return best_params, best_f1     
        else:
            for gamma in gammas:
                params['gamma'] = gamma
                results = np.concatenate([[self.cross_validation_per_k(k, params)] for k in range(self.num_folds)], axis=1)
                f1_scores.append(np.mean(results, axis=0)[-1])
            
            optimum_idx = np.argmax(f1_scores)
            best_params, best_f1 = gammas[optimum_idx], f1_scores[optimum_idx]
            return best_params, best_f1     
        

    def cross_validation_per_k(self, k: int, params: dict):
        """return the loss of given model."""
        # get k'th subgroup in test, others in train
        tr_indices, te_indices = self.k_indices[~(np.arange(self.k_indices.shape[0]) == k)].reshape(-1),\
                                 self.k_indices[k]

        # split the data based on train and validation indices
        y_trn, y_val = self.y[tr_indices], self.y[te_indices]
        x_trn, x_val = self.x[tr_indices], self.x[te_indices]
        x_stdr_trn, x_stdr_val = self.x_stdr[tr_indices], self.x_stdr[te_indices]

        x_trn_poly = build_poly(x_trn, self.degree)
        x_val_poly = build_poly(x_val, self.degree)

        x_trn_log = build_log_transformation(x_trn, np.arange(x_trn.shape[1]))
        x_val_log = build_log_transformation(x_val, np.arange(x_val.shape[1]))

        x_trn = np.c_[x_trn_poly, x_trn_log]
        x_val = np.c_[x_val_poly, x_val_log]

        x_trn = standardize(x_trn)
        x_val = standardize(x_val)

        ones_trn = np.ones([len(x_trn), 1])
        x_trn = np.c_[ones_trn, x_trn]
        #x_trn = np.c_[ones_trn,standardize(x_trn)]
        ones_val = np.ones([len(x_val), 1])
        x_val = np.c_[ones_val, x_val]
        #x_val = np.c_[ones_val, standardize(x_val)]
        # run the model
        #params['tx'], params['y'] = x_trn, y_trn
        params['y'], params['tx'] = y_trn, x_trn
        w, _ = self.model(**params)


        # calculate the loss for train and test data
        loss_trn = np.sqrt(mse_loss(y_trn, x_trn, w))
        loss_val = np.sqrt(mse_loss(y_val, x_val, w))
        
        # get validation f1-score
        y_pred = get_classification_pred(x_val, w)

        # check that there are no NaN values in the f1 score
        f1_val = f1_score(y_val, y_pred)
        if f1_val == NaN:
            f1_val = 0 

        return loss_trn, loss_val, f1_val

    def build_k_indices(self):
        """
        build k indices for k-fold.
        Args:
            y:      shape=(N,)
            k_fold: K in K-fold, i.e. the fold num
            seed:   the random seed

        Returns:
            A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
        """
        num_row  = self.y.shape[0]
        np.random.seed(self.num_seed)
        interval = int(num_row / self.num_folds)
        indices  = np.random.permutation(num_row)

        self.k_indices = np.array([indices[k * interval : (k + 1) * interval] for k in range(self.num_folds)])
   

