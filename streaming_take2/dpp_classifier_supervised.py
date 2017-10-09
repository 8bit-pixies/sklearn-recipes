import sklearn

from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_approximation import Nystroem
from dpp import sample_dpp, decompose_kernel, sample_conditional_dpp
from sklearn.preprocessing import normalize
import warnings

from sklearn import preprocessing
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np

import random
from collections import Counter

def class_separability(X, y):
    """
    Calculates the class separability based on the kernel paper
    """
    y = np.array(y)
    gamma = 1.0/X.shape[1]
    if X.shape[0] < 1000:
        L = rbf_kernel(X, gamma=gamma)
    else:
        L = Nystroem(gamma=gamma).fit_transform(X)
        L = L.dot(L.T)
    Ls = np.log(L)*(-1.0/(gamma))
    y_set = list(set(list(y)))
    
    N = L.shape[0]
    
    mask0 = y==y_set[0]
    mask1 = y==y_set[1]
    # print("X y")
    # print(X.shape)
    # print(y.shape)
    # print(rbf_kernel(X, gamma=gamma).shape)
    # print(rbf_kernel(X.T, gamma=gamma).shape)
    # print(Ls.shape)
    # print(mask0)
    # print(mask1)
    L_sel0 = Ls[mask0, :]
    L_sel0 = L_sel0.T.dot(L_sel0)
    L_sel1 = Ls[mask1, :]
    L_sel1 = L_sel1.T.dot(L_sel1)
    L_sel = np.sum(Ls[mask0, :][:, mask1]) + np.sum(Ls[mask1, :][:, mask0])
    n0 = L_sel0.shape[0]
    n1 = L_sel1.shape[0]
    # print("...")
    # print(y)
    # print(np.sum(L_sel0))
    # print(np.sum(L_sel1))
    s_w = (((1.0/n0) * (L_sel0)) + ((1.0/n1) * (L_sel1)))
    s_b = (1.0/(n0+n1))*L_sel
    return s_b, s_w

def evaluate_feats1(s_b, s_w, highest_best=True):
    curr_u1 = []
    curr_u2 = []
    my_feats = []
    prev_score = None
    X = s_b/s_w
    eval_order = np.argsort(X).flatten()
    if highest_best:
        eval_order = eval_order[::-1]
    for idx in list(eval_order):
        if prev_score is None:
            curr_u1.append(s_b[idx])
            curr_u2.append(s_w[idx])
            my_feats.append(idx)
        else:
            test_u1 = curr_u1[:]
            test_u2 = curr_u2[:]
            test_u1.append(s_b[idx])
            test_u2.append(s_w[idx])
            score = ((np.sum(test_u1)/np.sum(test_u2)) - prev_score)
            if score > 0.001:
                my_feats.append(idx)
                curr_u1.append(s_b[idx])
                curr_u2.append(s_w[idx])
        prev_score = np.sum(curr_u1)/np.sum(curr_u2)
    return list(my_feats)

def evaluate_feats2(X, alpha=0.05, highest_best=True):
    """
    X is the raw scrores
    alpha is the level of significance
    
    This version uses T-test
    
    Returns: set of indices indicating selected features.
    """
    eval_order = np.argsort(X)
    if highest_best:
        eval_order = eval_order[::-1]
    selected_feats = []
    selected_idx = []
    for idx in eval_order:
        if len(selected_feats) == 0:
            selected_feats.append(X[idx])
            selected_idx.append(idx)
            continue
        # now continue on and decide what to do
        mu = np.mean(selected_feats)
        sigma = np.std(selected_feats)
        U = len(selected_feats)
        if sigma == 0.0 and U > 1:
            return selected_idx
        elif sigma == 0.0:
            selected_feats.append(X[idx])
            selected_idx.append(idx)
            continue
        
        # otherwise compute score for T test.
        t_stat = (mu - X[idx])/(sigma/np.sqrt(U))
        t_alpha = stats.t.pdf(t_stat, U)
        if t_alpha <= alpha:
            selected_feats.append(X[idx])
            selected_idx.append(idx)
        else:
            return selected_idx
    return selected_idx

def evaluate_feats(s_b, s_w, alpha=0.05):
    set1 = evaluate_feats1(s_b,s_w)
    eval2 = s_b/s_w
    if len(eval2.shape) > 1:
        eval2 = np.diag(s_b)/np.diag(s_w)
    set2 = evaluate_feats2(eval2, alpha)
    return list(set(set1 + set2))

"""
Implement DPP version that is similar to what is done above


sketch of solution
------------------

DPP requires a known number of parameters to check at each partial fit!

"""

class DPPClassifier(SGDClassifier):
    def __init__(self, loss="log", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                 verbose=0, epsilon=0.1, n_jobs=1,
                 random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, class_weight=None, warm_start=False,
                 average=False, n_iter=None, 
                 intragroup_decay = 0.9, pca_alpha=0.05,
                 intragroup_alpha=0.05, intergroup_thres=None):
        super(DPPClassifier, self).__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, class_weight=class_weight, warm_start=warm_start,
            average=average, n_iter=n_iter)
        self.coef_info = {'cols': [], 'coef':[], 'excluded_cols': []}
        self.seen_cols = []
        self.base_shape = None
        self.dpp_k = {'pca': 0, 'kpca':0}
        self.unseen_only = False
        self.kernel = {'gamma': None, 'kernel': None} # this is the kernel that is used for DPP, unsupervised and supervised FS
        self.intragroup_alpha = intragroup_alpha
        self.intergroup_thres = intergroup_thres if intergroup_thres is not None else epsilon
    
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
        col_coef = [(col, coef) for col, coef in zip(X.columns.tolist(), self.coef_.flatten()) if np.abs(coef) >= self.intergroup_thres]
        self.coef_info['cols'] = [x for x, _ in col_coef]
        self.coef_info['coef'] = [x for _, x in col_coef]
        self.coef_info['excluded_cols'] = [x for x in self.seen_cols if x not in self.coef_info['cols']]
        self.coef_ = np.array(self.coef_info['coef']).reshape(1, -1)
    
    def _dpp_sel(self, X_, y=None):
        """
        DPP only relies on X. 
        
        We will condition the sampling based on:
        *  `self.coef_info['cols']`
        
        After sampling it will go ahead and then perform grouped wilcoxon selection.
        """
        X = np.array(X_)
        cols_to_index = [idx for idx, x in enumerate(X_.columns) if x in self.coef_info['cols']]
        unseen_cols_to_index = [idx for idx, x in enumerate(X_.columns) if x not in self.coef_info['cols']]
        gamma = 1.0/X.T.shape[1]
        if X.shape[1] < 1000:
            feat_dist = rbf_kernel(X.T, gamma=gamma)
        else:
            feat_dist = Nystroem(gamma=gamma).fit_transform(X.T)
            feat_dist = feat_dist.dot(feat_dist.T)
        
        self.kernel['gamma'] = gamma
        self.kernel['kernel'] = feat_dist.copy()
        k = None
        print("\tSampling DPP...")
        if len(self.coef_info['cols']) == 0:
            feat_index = sample_dpp(decompose_kernel(feat_dist), k=k)
            feat_index = [x for x in feat_index if x is not None]
        else:            
            feat_index = sample_conditional_dpp(feat_dist, cols_to_index, k=k)
            feat_index = [x for x in feat_index if x is not None]
        print("\tSampling DPP Done!")
        
        print("\tCalculating separability (covariance matrix)...")
        s_b, s_w = class_separability(X, y)
        col_sel = evaluate_feats(s_b, s_w)
        print("\tCalculating separability Done!")
        
        self.unseen_only = False # perhaps add more conditions around unseen - i.e. once unseen condition kicks in, it remains active?
        self.coef_info['cols'] = list(set(self.coef_info['cols'] + col_sel))
        col_rem = X_.columns.difference(self.coef_info['cols'])
        # update column exclusion...
        self.coef_info['excluded_cols'] = [x for x in self.coef_info['excluded_cols'] if x not in self.coef_info['cols']]
        self.add_column_exclusion(col_rem)
        
    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        
        # TODO: add DPP selection
        self.coef_info = {'cols': [], 'coef':[], 'excluded_cols': []}
        #self._dpp_sel(X, y)
        #X = self._fit_columns(X)
        
        super(DPPClassifier, self).fit(X, y, coef_init=coef_init, intercept_init=intercept_init,
            sample_weight=sample_weight)
        self._reg_penalty(X)
        return self
    
    def partial_fit(self, X, y, sample_weight=None):
        X_ = X.copy()
        print(X.shape)
        unseen_col_size = len([1 for x in X.columns if x not in self.seen_cols])
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        #sample_from_exclude_size = int(len(self.coef_info['excluded_cols']) - (len(self.coef_info['cols'])/2.0))+1
        sample_from_exclude_size = int(len(self.coef_info['excluded_cols']) - unseen_col_size)
        if sample_from_exclude_size > 0:
            cols_excl_sample = random.sample(self.coef_info['excluded_cols'], sample_from_exclude_size)
            X = X[X.columns.difference(cols_excl_sample)]
        #X = X[X.columns.difference(self.coef_info['excluded_cols'])]
        
        # TODO: add DPP selection
        self._dpp_sel(X, y)
        X = self._fit_columns(X_)
        
        # now update coefficients
        n_samples, n_features = X.shape
        coef_list = np.zeros(n_features, dtype=np.float64, order="C")
        coef_list[:len(self.coef_info['coef'])] = self.coef_info['coef']
        self.coef_ = np.array(coef_list).reshape(1, -1)
        
        super(DPPClassifier, self).partial_fit(X, y, sample_weight=None)  
        self._reg_penalty(X)
        return self
    
    def predict(self, X):
        X = self._fit_columns(X, transform_only=True)
        return super(DPPClassifier, self).predict(X)
    def predict_proba(self, X):
        X = self._fit_columns(X, transform_only=True)
        return super(DPPClassifier, self).predict_proba(X)