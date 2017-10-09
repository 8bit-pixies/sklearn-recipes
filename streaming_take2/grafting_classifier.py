import sklearn

from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

import pandas as pd
import numpy as np

class GraftingClassifier(SGDClassifier):
    def __init__(self, loss="log", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                 verbose=0, epsilon=0.1, n_jobs=1,
                 random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, class_weight=None, warm_start=False,
                 average=False, n_iter=None, 
                 reg_penalty=None):
        super(GraftingClassifier, self).__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, class_weight=class_weight, warm_start=warm_start,
            average=average, n_iter=n_iter)
        self.coef_info = {'cols': [], 'coef':[], 'excluded_cols': []}
        self.seen_cols = []
        self.base_shape = None
        self.reg_penalty = reg_penalty if reg_penalty is not None else l1_ratio
        
    def add_column_exclusion(self, cols):
        self.coef_info['excluded_cols'] = list(self.coef_info['excluded_cols']) + list(cols)
    
    def _fit_columns(self, X_, return_x=True, transform_only=False):
        """
        Method filter through "unselected" columns. The goal of this 
        method is to filter any uninformative columns.
        
        This will be selected based on index only?
        
        If return_x is false, it will only return the boolean mask.
        """
        X = X_[X_.columns.difference(self.coef_info['excluded_cols'])]
        
        # order the columns correctly...
        col_order = self.coef_info['cols'] + list([x for x in X.columns if x not in self.coef_info['cols']])
        X = X[col_order]
        return X
        
    def _reg_penalty(self, X):
        col_coef = [(col, coef) for col, coef in zip(X.columns.tolist(), self.coef_.flatten()) if np.abs(coef) >= self.reg_penalty]
        self.coef_info['cols'] = [x for x, _ in col_coef]
        self.coef_info['coef'] = [x for _, x in col_coef]
        self.coef_info['excluded_cols'] = [x for x in self.seen_cols if x not in self.coef_info['cols']]
        self.coef_ = np.array(self.coef_info['coef']).reshape(1, -1) 
            
    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        super(GraftingClassifier, self).fit(X, y, coef_init=coef_init, intercept_init=intercept_init,
            sample_weight=sample_weight)
        self._reg_penalty(X)
        return self
        
    def partial_fit(self, X, y, sample_weight=None):
        X_ = X.copy()
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        X = X[X.columns.difference(self.coef_info['excluded_cols'])]
        
        # TODO: add the spectral selection here
        # it should only consider "unseen"
        X = self._fit_columns(X)        
        # now update coefficients
        n_samples, n_features = X.shape
        coef_list = np.zeros(n_features, dtype=np.float64, order="C")
        coef_list[:len(self.coef_info['coef'])] = self.coef_info['coef']
        self.coef_ = np.array(coef_list).reshape(1, -1)
        
        super(GraftingClassifier, self).partial_fit(X, y, sample_weight=None)  
        self._reg_penalty(X)
        return self
    
    def predict(self, X):
        X = self._fit_columns(X, transform_only=True)
        return super(GraftingClassifier, self).predict(X)
        
    def predict_proba(self, X):
        X = self._fit_columns(X, transform_only=True)
        return super(GraftingClassifier, self).predict_proba(X)