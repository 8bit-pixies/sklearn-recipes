# osfs_classifier

import sklearn

from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import SGDRegressor, SGDClassifier

import pandas as pd
import numpy as np

from osfs_util import dependence_test, fisher_test
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import BayesianGaussianMixture

import pandas

class OSFSClassifier(SGDClassifier):
    def __init__(self, loss="log", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                 verbose=0, epsilon=0.1, n_jobs=1,
                 random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, class_weight=None, warm_start=False,
                 average=False, n_iter=None, 
                 relevance_alpha=0.05):
        super(OSFSClassifier, self).__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, class_weight=class_weight, warm_start=warm_start,
            average=average, n_iter=n_iter)
        """
        relevance_alpha: the alpha level for the conditional independence statistics tests
        """
        self.coef_info = {'cols': [], 'coef':[], 'excluded_cols': [], 
                          'strong_dep': [], 'weak_dep': []}
        self.seen_cols = []
        self.base_shape = None
        self.relevance_alpha = relevance_alpha
    
    def add_column_exclusion(self, cols):
        self.coef_info['excluded_cols'] = self.coef_info['excluded_cols'] + cols
        
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
    
    def _redundancy(self, X, y):
        _, pval_m = fisher_test(np.array(X), y)
        col_weak = []
        redun_cols = []
        for idx, colname in enumerate(X.columns.tolist()):
            if colname not in self.coef_info['weak_dep']:
                continue
            # determine redudundancy.
            pvals = pval_m[idx, :]
            redun_alpha = np.min(pvals)
            if redun_alpha < self.relevance_alpha:
                col_weak.append(colname)
            else:
                redun_cols.append(colname)
        col_coef = [(col, coef) for col, coef in zip(X.columns.tolist(), self.coef_.flatten()) if col in col_weak]
        self.coef_info['cols'] = [x for x, _ in col_coef]
        self.coef_info['coef'] = [x for _, x in col_coef]
        
        # excl cols that are in self.coef_info['weak_dep']
        weak_dep = [x for x in self.coef_info['weak_dep'][:] if x in col_weak]
        self.coef_info['weak_dep'] = weak_dep[:]
        self.coef_info['excluded_cols'] = [x for x in self.seen_cols if x not in self.coef_info['cols']]
        self.coef_ = np.array(self.coef_info['coef']).reshape(1, -1)
    
    def _osfs_sel(self, X_, y):
        """
        Partial fit online group feature selection method to 
        perform spectral analysis on incoming feature set
        to then expand the coefficient listing
        """
        X = np.array(X_)
        _, pval_m = fisher_test(X, y)
        cols_to_index = [(idx, x) for idx, x in enumerate(X_.columns) if x in self.coef_info['cols']]
        unseen_cols_to_index = [(idx, x) for idx, x in enumerate(X_.columns) if x not in self.coef_info['cols']]
        # get the appropriate submatrix
        
        col_strong = []
        col_weak = []
        #print(X_.columns.tolist())
        # we will have to evaluate each feature one at a time!
        for new_col, colname in unseen_cols_to_index:
            new_sel = cols_to_index + col_strong + col_weak + [colname]
            colname_to_indx = [idx for idx, x in enumerate(X_.columns.tolist()) if x in new_sel]
            #print(colname_to_indx)
            if len(colname_to_indx) == 1:
                col_weak.append(colname)
                continue
            pval_test = pval_m[colname_to_indx, :][:, colname_to_indx]
            try:
                strong_dep = np.max(pval_test[np.triu_indices(pval_test.shape[0], 1)])
                weak_dep   = np.min(pval_test[np.triu_indices(pval_test.shape[0], 1)])
            except:
                strong_dep = float("inf")
                weak_dep = float("inf")
            #print(strong_dep)
            #print(weak_dep)
            if strong_dep < self.relevance_alpha:
                col_strong.append(colname)
            elif weak_dep < self.relevance_alpha:
                col_weak.append(colname)
        
        # now evaluate all the weak columns to determine if they are redundant or not.
        """
        sel_cols = list(self.coef_info['cols']) + list(col_strong) + list(col_weak)
        X_redun = X_[sel_cols]
        if X_redun.shape[0] > 0:
            _, pval_m = fisher_test(np.array(X_redun), y)
            for idx, col in enumerate(X_redun.columns):
                if col not in col_weak:
                    continue
                elif np.min(pval_m[idx, :]) < self.relevance_alpha:
                    col_strong.append(col)
        """
        # update infomation
        #print(self.coef_info['cols'])
        #print(col_strong)
        self.coef_info['cols'] = list(set(self.coef_info['cols'] + col_strong + col_weak))
        self.coef_info['strong_dep'] = list(set(self.coef_info['strong_dep'] + col_strong))
        self.coef_info['weak_dep'] = list(set(self.coef_info['weak_dep'] + col_weak))
        self.coef_info['excluded_cols'] = [col for col in self.seen_cols if col not in self.coef_info['cols']]        
        
    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        X_ = X.copy()
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        
        # TODO: add the spectral selection here
        self._osfs_sel(X, y)
        #self.coef_info['weak_dep'] = X.columns.tolist()
        X = self._fit_columns(X)
        
        super(OSFSClassifier, self).fit(X, y, coef_init=coef_init, intercept_init=intercept_init,
            sample_weight=sample_weight)
        self._redundancy(X, y)
        return self
    
    def partial_fit(self, X, y, sample_weight=None):
        X_ = X.copy()
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        X = X[X.columns.difference(self.coef_info['excluded_cols'])]
        #print(X.shape)
        self._osfs_sel(X, y)
        X = self._fit_columns(X)
        #print(X.shape)
        
        # now update coefficients
        n_samples, n_features = X.shape
        coef_list = np.zeros(n_features, dtype=np.float64, order="C")
        coef_list[:len(self.coef_info['coef'])] = self.coef_info['coef']
        self.coef_ = np.array(coef_list).reshape(1, -1)
        super(OSFSClassifier, self).partial_fit(X, y, sample_weight=None)  
        self._redundancy(X, y)
        return self
    
    def predict(self, X):
        X = self._fit_columns(X, transform_only=True)
        #print(X.shape)
        return super(OSFSClassifier, self).predict(X)
    def predict_proba(self, X):
        X = self._fit_columns(X, transform_only=True)
        #print(X.shape)
        return super(OSFSClassifier, self).predict_proba(X)   