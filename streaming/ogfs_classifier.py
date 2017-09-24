import sklearn

from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import SGDRegressor, SGDClassifier

import pandas as pd
import numpy as np

import SPEC
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.mixture import BayesianGaussianMixture


def similarity_within_class(X, y):
    return SPEC.similarity_classification(X, y)

def similarity_between_class(X, y):
    """
    Calculates betweenclass affinity X (data) y (labels)
    
    note that it only considers the labels
    """
    y_series = pd.Series(y)
    y_val = y_series.value_counts(normalize=True)
    n_inv = 1.0/len(set(y))
    
    y_size = len(y)
    sim_matrix = np.zeros((len(y), len(y)))
    for s_i in range(y_size):
        for s_j in range(y_size):
            sim_matrix[s_i, s_j] = n_inv - y_val[y[s_i]] if y[s_i] == y[s_j] else n_inv
    return sim_matrix

def convert_to_deciles(y, n=10, gmm=False):
    """
    By default converts to deciles, can be changed based on choice of n.
    """
    if gmm:
        # this is experimental
        bgm = BayesianGaussianMixture(n_components=10)
        bgm.fit(y.reshape(-1, 1))
        return bgm.predict(y.reshape(-1, 1))
    return np.array(pd.cut(y, n, labels=range(n)))

def spec_supervised(X, y, is_classification=True):
    if not is_classification:
        y = convert_to_deciles(y, 10, gmm=False)
    # sample X if it is too big...
    instances_count = X.shape[0]
    if instances_count > 1000:
        idx = np.random.randint(instances_count, size=1000)
        X = X[idx, :]
    W_w = similarity_within_class(X, y)
    W_b = similarity_between_class(X, y)
    s_w = SPEC.spec(**{'X': X, 'y': y, 'style':0, 'mode': 'raw', 'W': W_w})
    s_b = SPEC.spec(**{'X': X, 'y': y, 'style':0, 'mode': 'raw', 'W': W_b})
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
    set2 = evaluate_feats2(s_b/s_w, alpha)
    return list(set(set1 + set2))

import pandas

class OGFSClassifier(SGDClassifier):
    def __init__(self, loss="log", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=None, tol=None, shuffle=True,
                 verbose=0, epsilon=0.1, n_jobs=1,
                 random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, class_weight=None, warm_start=False,
                 average=False, n_iter=None, 
                 intragroup_alpha=0.05, intergroup_thres=None):
        super(OGFSClassifier, self).__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, class_weight=class_weight, warm_start=warm_start,
            average=average, n_iter=n_iter)
        """
        intragroup_alpha : the alpha level of t-test used to determine significance
        intergroup_thres : the threshold for lasso to remove redundancy
        """
        self.coef_info = {'cols': [], 'coef':[], 'excluded_cols': []}
        self.seen_cols = []
        self.base_shape = None
        self.intragroup_alpha = intragroup_alpha
        self.intergroup_thres = intergroup_thres if intergroup_thres is not None else epsilon
    
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
    
    def _reg_penalty(self, X):
        col_coef = [(col, coef) for col, coef in zip(X.columns.tolist(), self.coef_.flatten()) if np.abs(coef) >= self.intergroup_thres]
        self.coef_info['cols'] = [x for x, _ in col_coef]
        self.coef_info['coef'] = [x for _, x in col_coef]
        self.coef_info['excluded_cols'] = [x for x in self.seen_cols if x not in self.coef_info['cols']]
        self.coef_ = np.array(self.coef_info['coef']).reshape(1, -1) 
            
    def _spectral_sel(self, X_, y):
        """
        Partial fit online group feature selection method to 
        perform spectral analysis on incoming feature set
        to then expand the coefficient listing
        """
        X = np.array(X_)        
        s_b, s_w = spec_supervised(X, y, True)
        col_sel = X_.columns[evaluate_feats(s_b, s_w)]
        sel_cols = list(self.coef_info['cols']) + list(col_sel)
        # update removed columns
        self.coef_info['excluded_cols'] = [col for col in self.seen_cols if col not in sel_cols]
        
        
    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        X_ = X.copy()
        self.coef_info = {'cols': [], 'coef':[], 'excluded_cols': []}
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        self.base_shape = None
        # TODO: add the spectral selection here
        self._spectral_sel(X, y)
        X = self._fit_columns(X)
        
        super(OGFSClassifier, self).fit(X, y, coef_init=coef_init, intercept_init=intercept_init,
            sample_weight=sample_weight)
        self._reg_penalty(X)
        return self
    
    def partial_fit(self, X, y, sample_weight=None):
        X_ = X.copy()
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        X = X[X.columns.difference(self.coef_info['excluded_cols'])]
        
        # TODO: add the spectral selection here
        # it should only consider "unseen"
        self._spectral_sel(X[X.columns.difference(self.coef_info['cols'])], y)
        X = self._fit_columns(X)
        
        # now update coefficients
        n_samples, n_features = X.shape
        coef_list = np.zeros(n_features, dtype=np.float64, order="C")
        coef_list[:len(self.coef_info['coef'])] = self.coef_info['coef']
        self.coef_ = np.array(coef_list).reshape(1, -1)
        
        super(OGFSClassifier, self).partial_fit(X, y, sample_weight=None)  
        self._reg_penalty(X)
        return self
    
    def predict(self, X):
        X = self._fit_columns(X, transform_only=True)
        return super(OGFSClassifier, self).predict(X)
    def predict_proba(self, X):
        X = self._fit_columns(X, transform_only=True)
        return super(OGFSClassifier, self).predict_proba(X)


