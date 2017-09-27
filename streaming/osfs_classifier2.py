# osfs_classifier

import sklearn

from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import SGDRegressor, SGDClassifier

from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import SGDRegressor, SGDClassifier

import pandas as pd
import numpy as np

from osfs_util import partial_dep_test
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import BayesianGaussianMixture
import random
import pandas

class OSFSClassifier(SGDClassifier):
    def __init__(self, loss="log", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                 verbose=0, epsilon=0.1, n_jobs=1,
                 random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, class_weight=None, warm_start=False,
                 average=False, n_iter=None, 
                 relevance_alpha=0.05, fast_osfs=True):
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
        self.partial_info = []
        self.base_shape = None
        self.relevance_alpha = relevance_alpha
        self.mode = 'weak_only' if fast_osfs else 'all'
        self.fast_osfs = True
    
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
    
    def _redundancy(self, X, y, mode='weak_only'): 
        col_redun = []
        X_temp = X[self.coef_info['cols']]
        if mode == 'weak_only':
            col_eval = self.coef_info['weak_dep']
        else:
            col_eval = self.coef_info['cols']
        for col in col_eval:            
            x_dat = X_temp[X_temp.columns.difference([col]+col_redun)]
            x1 = np.array(X[[col]]).flatten()
            if mode == 'weak_only':
                #print("\t\t{}".format(x_dat.shape))
                partial_cor = partial_dep_test(x1, y, np.array(x_dat), col, list(x_dat.columns), 1, prev=self.partial_info[:], early_stopping=True, alpha=self.relevance_alpha)
            else:
                print("\t\t{}".format(x_dat.shape))
                depth = 1 if x_dat.shape[1] < 500 else 1
                partial_cor = partial_dep_test(x1, y, np.array(x_dat), col, list(x_dat.columns), depth, prev=self.partial_info[:], early_stopping=True, alpha=self.relevance_alpha)
            self.partial_info = self.partial_info[:] + partial_cor
            #print(x1)
            #print(y)
            #print(x_dat.shape)
            #print(col)
            #print(list(x_dat.columns))
            #print(partial_cor)
            #print("iter: {}".format(col))
            test_case = [x['pval'] for x in partial_cor]
            if len(test_case) == 0:
                continue
            strong_dep = np.max([x['pval'] for x in partial_cor])
            if strong_dep >= self.relevance_alpha:
                col_redun.append(col)
        self.partial_info = self.partial_info[:1000]
        # excl cols that are in self.coef_info['weak_dep']
        col_coef = [(col, coef) for col, coef in zip(X.columns.tolist(), self.coef_.flatten()) if col not in col_redun]        
        self.coef_info['cols'] = [x for x, _ in col_coef]
        self.coef_info['coef'] = [x for _, x in col_coef]
        self.coef_info['strong_dep'] = self.coef_info['cols'][:]
        self.coef_info['weak_dep'] = []
        self.coef_info['excluded_cols'] = [x for x in self.seen_cols if x not in self.coef_info['cols']]
        self.coef_ = np.array(self.coef_info['coef']).reshape(1, -1)
    
    def _osfs_sel(self, X_, y):
        """
        Partial fit online group feature selection method to 
        perform spectral analysis on incoming feature set
        to then expand the coefficient listing
        """
        X = np.array(X_)
        cols_to_index = [(idx, x) for idx, x in enumerate(X_.columns) if x in self.coef_info['cols']]
        unseen_cols_to_index = [(idx, x) for idx, x in enumerate(X_.columns) if x not in self.coef_info['cols']]
        
        # iterate to determine strong/weak relevance
        cols_name = [x[1] for x in cols_to_index]
        col_strong = []
        col_weak = []
        x_data = np.array(X_[cols_name])
        
        if not self.fast_osfs:
            for new_col, colname in unseen_cols_to_index:
                x_data = np.array(X_[cols_name+col_strong+col_weak])
                if x_data.shape[1] == 0:
                    col_weak.append(colname)
                    continue
                x1 = np.array(X_[[colname]]).flatten()
                if len(set(list(x1))) == 1:
                    # remove constant fields (we can add variance filter later...)
                    continue
                partial_cor = partial_dep_test(x1, y, x_data, colname, cols_name+col_strong+col_weak, 3, prev=self.partial_info[:])
                self.partial_info = self.partial_info[:] + partial_cor  
                # perform dependency check with no conditioning for fast osfs
                #print(x1)
                #print(y)
                #print(x_data.shape)
                #print(colname)
                #print(cols_name+col_strong+col_weak)
                #print("iter: {}".format(colname))
                
                strong_dep = np.max([x['pval'] for x in partial_cor])
                weak_dep = np.min([x['pval'] for x in partial_cor])
                if strong_dep < self.relevance_alpha:
                    col_strong.append(colname)
                elif weak_dep < self.relevance_alpha:
                    col_weak.append(colname)
        else:
            # simply perform one-way analysis
            unseen_col = [y for x, y in unseen_cols_to_index]
            x_dat_unseen = X_[unseen_col]
            _, f_pval = sklearn.feature_selection.f_classif(x_dat_unseen, y)
            col_weak = [x for x, y in list(zip(unseen_col, f_pval)) if y < self.relevance_alpha and not np.isnan(y)]
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
        no_redundancy=False
        if X.shape[1] == 0:
            # force it to add all columns for now...
            no_redundancy=True
            self.coef_info['cols'] = self.seen_cols[:]
            self.coef_info['strong_dep'] = self.coef_info['cols'][:]
            self.coef_info['weak_dep'] = []
            self.coef_info['excluded_cols'] = [x for x in self.seen_cols if x not in self.coef_info['cols']]
            X = X_.copy()
            X = self._fit_columns(X)
        
        super(OSFSClassifier, self).fit(X, y, coef_init=coef_init, intercept_init=intercept_init,
            sample_weight=sample_weight)
        if no_redundancy:
            self._redundancy(X, y, self.mode)
        return self
    
    def partial_fit(self, X, y, sample_weight=None):
        X_ = X.copy()
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        X = X[X.columns.difference(self.coef_info['excluded_cols'])]
        #print(X.shape)
        self._osfs_sel(X, y)
        X = self._fit_columns(X)
        no_redundancy=False
        if X.shape[1] == 0:
            # force it to add all columns for now...
            no_redundancy=True
            self.coef_info['cols'] = self.seen_cols[:]
            self.coef_info['strong_dep'] = self.coef_info['cols'][:]
            self.coef_info['weak_dep'] = []
            self.coef_info['excluded_cols'] = [x for x in self.seen_cols if x not in self.coef_info['cols']]
            X = X_.copy()
            X = self._fit_columns(X)
        
        # now update coefficients
        n_samples, n_features = X.shape
        coef_list = np.zeros(n_features, dtype=np.float64, order="C")
        coef_list[:len(self.coef_info['coef'])] = self.coef_info['coef']
        self.coef_ = np.array(coef_list).reshape(1, -1)
        super(OSFSClassifier, self).partial_fit(X, y, sample_weight=None)  
        if no_redundancy:
            self._redundancy(X, y, 'weak_only')
            if self.mode != 'weak_only':
                self._redundancy(X, y, self.mode)
        return self
    
    def predict(self, X):
        X = self._fit_columns(X, transform_only=True)
        #print(X.shape)
        return super(OSFSClassifier, self).predict(X)
    def predict_proba(self, X):
        X = self._fit_columns(X, transform_only=True)
        #print(X.shape)
        return super(OSFSClassifier, self).predict_proba(X)