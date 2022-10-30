"""Cross Validation methods to optimize hyperparameters"""

#from scripts.feature_expansion import build_log_transformation
import numpy as np
from feature_expansion import *
from implementations import *
from itertools import product
from typing import Tuple
from metrics import *



class HyperParameterTuner:
    def __init__(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        model_name: str,
        num_folds: int,
        num_seed: int=0,
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
        self.num_seed = num_seed
        self.num_folds = num_folds
        self.max_iter = max_iter
        self.model_name = model_name
        
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

        self.degree = np.arange(5)

    def tune_(self) -> Tuple[list, float]:
        """
        hyperparameter tuning done by grid search.
        best parameters are found by finding the maximum f1 scores.
        """
        lambdas = np.logspace(-15,0,15)
        gammas  = np.linspace(0,1,100)        

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
            for lambda_ in lambdas:
                params['lambda_'] =lambda_
                results = np.concatenate([[self.cross_validation_per_k(k, params)] for k in range(self.num_folds)], axis=1)
                f1_scores.append(np.mean(results, axis=0)[-1])
                print("ended lambda", lambda_)
                        
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

        x_trn = build_poly(x_trn, self.degree)
        x_val = build_poly(x_val, self.degree)

        x_trn_log = build_log_transformation(x_trn)
        x_val_log = build_log_transformation(x_val)

        x_trn = np.c_[x_trn, x_trn_log]
        x_val = np.c_[x_val, x_val_log]

        # run the model
        params['tx'], params['y'] = x_trn, y_trn
        w, _ = self.model(**params)
        
        # calculate the loss for train and test data
        loss_trn = np.sqrt(mse_loss(y_trn, x_trn, w))
        loss_val = np.sqrt(mse_loss(y_val, x_val, w))
        
        # get validation f1-score
        y_pred = get_classification_pred(x_val, w)
        f1_val = f1_score(y_val, y_pred)
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
        np.random.seed()
        interval = int(num_row / self.num_folds)
        indices  = np.random.permutation(num_row)

        self.k_indices = np.array([indices[k * interval : (k + 1) * interval] for k in range(self.num_folds)])
   

