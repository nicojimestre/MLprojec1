"""Cross Validation methods to optimize hyperparameters"""

import numpy as np
from itertools import product
from metrics import f1_score
from implementations import *
from feature_expansion import *


class HyperParameterTuner:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model_name: str,
        num_folds: int = 5,
        num_seed: int = 0,
        max_iter: int = 10,
        grid_size: int = 10,
    ):

        available_models = {
            "least_squares": least_squares,
            "ridge": ridge_regression,
            "mse_gd": mean_squared_error_gd,
            "mse_sgd": mean_squared_error_sgd,
            "logistic": logistic_regression,
            "reg_logistic": reg_logistic_regression,
        }

        self.x = x
        self.y = y
        self.model = available_models[model_name]
        self.model_name = model_name
        self.num_folds = num_folds
        self.num_seed = num_seed
        self.max_iter = max_iter
        self.grid_size = grid_size

        # build k_indices
        np.random.seed(self.num_seed)
        self.build_k_indices()

        # get model params given model specs.
        model_parameters = {
            "least_squares": {},
            "ridge": {"lambda_": None},
            "mse_gd": {
                "initial_w": np.ones((x.shape[1],)),
                "max_iters": self.max_iter,
                "gamma": None,
            },
            "mse_sgd": {
                "initial_w": np.ones((x.shape[1],)),
                "max_iters": self.max_iter,
                "gamma": None,
            },
            "logistic": {
                "initial_w": np.ones((x.shape[1],)),
                "max_iters": self.max_iter,
                "gamma": None,
            },
            "reg_logistic": {
                "initial_w": np.ones((x.shape[1],)),
                "max_iters": self.max_iter,
                "gamma": None,
                "lambda_": None,
            },
        }
        self.hyp_params = model_parameters[model_name]

    def tune_(self) -> Tuple[list, float]:
        """
        hyperparameter tuning done by grid search.
        best parameters are found by finding the maximum f1 scores.
        """

        lambdas = np.logspace(-15, 2, self.grid_size)
        gammas = np.logspace(-6, -0.25, self.grid_size)

        f1_scores = np.array([])
        params = self.hyp_params

        # if model is least squares
        if self.model_name == "least_squares":
            results = np.array(
                [
                    [k, self.cross_validation_per_k(k, self.hyp_params)[-1]]
                    for k in range(self.num_folds)
                ]
            )
            return results[np.argmax(results[:, -1])]

        # if model is regularized ridge regression
        elif self.model_name == "reg_logistic":
            lambda_and_gammas = list(product(gammas, lambdas))
            for i, (gamma, lambda_) in enumerate(lambda_and_gammas):
                params["gamma"], params["lambda_"] = gamma, lambda_
                results = np.array(
                    [
                        self.cross_validation_per_k(k, params)
                        for k in range(self.num_folds)
                    ]
                )
                f1_scores = np.append(f1_scores, np.mean(results, axis=0)[-1])
                if i % 10 == 0:
                    print(
                        f"Progress {i}/{self.grid_size**2}, lambda_: {lambda_},  gamma: {gamma}, f1: {np.round(np.mean(results, axis=0)[-1], 4)}"
                    )

            f1_scores[np.isnan(f1_scores)] = 0
            optimum_idx = np.argmax(f1_scores)
            print(optimum_idx, len(f1_scores), len(list(lambda_and_gammas)))
            best_params, best_f1 = (
                lambda_and_gammas[optimum_idx],
                f1_scores[optimum_idx],
            )
            return {"params": best_params, "f1_score": best_f1}

        # if model is ridge regression
        elif self.model_name == "ridge":
            for i, lambda_ in enumerate(lambdas):
                params["lambda_"] = lambda_
                results = np.array(
                    [
                        self.cross_validation_per_k(k, params)
                        for k in range(self.num_folds)
                    ]
                )
                f1_scores = np.append(f1_scores, np.mean(results, axis=0)[-1])
                if i % 10 == 0:
                    print(
                        f"Progress {i}/{len(lambdas)}, lambda_: {lambda_}, f1: {np.round(np.mean(results, axis=0)[-1], 4)}"
                    )

            f1_scores[np.isnan(f1_scores)] = 0
            optimum_idx = np.argmax(f1_scores)
            best_params, best_f1 = lambdas[optimum_idx], f1_scores[optimum_idx]
            return {"lambda_": best_params, "f1_score": best_f1}

        # do otherwise (mse_gd, mse_sgd, logistic)
        else:
            for i, gamma in enumerate(gammas):
                params["gamma"] = gamma
                results = np.array(
                    [
                        self.cross_validation_per_k(k, params)
                        for k in range(self.num_folds)
                    ]
                )
                f1_scores = np.append(f1_scores, np.mean(results, axis=0)[-1])
                if i % 10 == 0:
                    print(
                        f"Progress {i}/{len(gammas)}, gamma :{gamma} f1: {np.round(np.mean(results, axis=0)[-1], 4)}"
                    )
            # set nan values to 0.
            f1_scores[np.isnan(f1_scores)] = 0

            # get optimum index
            optimum_idx = np.argmax(f1_scores)
            best_params, best_f1 = gammas[optimum_idx], f1_scores[optimum_idx]
            return {"gamma": best_params, "f1_score": best_f1}

    def cross_validation_per_k(self, k: int, params: dict) -> list:
        """return the loss of given model."""
        # get k'th subgroup in test, others in train
        tr_indices, te_indices = (
            self.k_indices[~(np.arange(self.k_indices.shape[0]) == k)].reshape(-1),
            self.k_indices[k],
        )

        # split the data based on train and validation indices
        y_trn, y_val = self.y[tr_indices], self.y[te_indices]
        x_trn, x_val = self.x[tr_indices], self.x[te_indices]

        # run the model
        params["tx"], params["y"] = x_trn, y_trn
        w, _ = self.model(**params)

        # calculate the loss for train and test data
        loss_trn = np.sqrt(mse_loss(y_trn, x_trn, w))
        loss_val = np.sqrt(mse_loss(y_val, x_val, w))

        # get validation f1-score
        y_pred = get_classification_pred(x_val, w)
        f1_val = f1_score(y_val, y_pred)
        return [loss_trn, loss_val, f1_val]

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
        num_row = self.y.shape[0]
        interval = int(num_row / self.num_folds)
        indices = np.random.permutation(num_row)

        self.k_indices = np.array(
            [indices[k * interval : (k + 1) * interval] for k in range(self.num_folds)]
        )
