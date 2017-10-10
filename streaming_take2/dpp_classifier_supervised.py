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

import random
from collections import Counter

def fast_euclid(X):
    gamma = 1.0/X.shape[1]
    if X.shape[0] < 1000:
        L = rbf_kernel(X, gamma=gamma)
    else:
        L = Nystroem(gamma=gamma).fit_transform(X)
        L = L.dot(L.T)
    Ls = np.log(L)*(-1.0/(gamma))
    return Ls
    
def class_separability(X, y, mode='mitra'):
    """
    Calculates the class separability based on the mitra paper    
    """    
    # get prior probs
    prior_proba = Counter(y)
    
    s_w = []    
    s_b = []
    m_o = np.mean(X, axis=0).reshape(-1, 1)
    
    if X.shape[0] > 1000:
        mode = 'kernel'
    
    for class_ in prior_proba.keys():
        mask = y==class_
        X_sel = X[mask, :]
        if mode == 'mitra':
            cov_sig = np.cov(X_sel.T)
            s_w.append(cov_sig * prior_proba[class_])
        else:
            K = fast_euclid(X_sel.T)
            s_w.append(K * prior_proba[class_])
        mu_m = prior_proba[class_] - m_o
        s_b.append(np.dot(mu_m, mu_m.T))
    s_w = np.atleast_2d(np.add(*s_w))
    s_b = np.add(*s_b)
    return s_b, s_w

def evaluate_feats0(s_b, s_w):
    curr_u1 = []
    curr_u2 = []
    my_feats = []
    prev_score = None
    try:
        s_b_inv = np.linalg.inv(s_b)
    except:
        s_b_inv = np.linalg.pinv(s_b)
    S = np.trace(np.dot(s_b_inv, s_w))
    eval_order = np.argsort(S).flatten()
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
            score = (prev_score - (np.sum(test_u1)/np.sum(test_u2)))
            if score > 0.001:
                my_feats.append(idx)
                curr_u1.append(s_b[idx])
                curr_u2.append(s_w[idx])
        prev_score = np.sum(curr_u1)/np.sum(curr_u2)
    return list(my_feats)

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

def entropy(X):
    mm = MinMaxScaler()
    X_mm = mm.fit_transform(X)
    Dpq = euclidean_distances(X_mm)
    D_bar = np.mean([x for x in np.triu(Dpq).flatten() if x != 0])
    alpha = -np.log(0.5)/D_bar
    sim_pq = np.exp(-alpha * Dpq)
    log_sim_pq = np.log(sim_pq)
    entropy = -2*np.sum(np.triu(sim_pq*log_sim_pq + ((1-sim_pq)*np.log((1-sim_pq))), 1))
    return entropy

def wilcoxon_group(X, f):
    """
    Wilcoxon is a very aggressive selector in an unsupervised sense. 
    Do we require a supervised group selection? (probably)
    
    Probably one that is score based in order to select the "best" ones
    similar to OGFS?
    """
    # X is a matrix, f is a single vector
    if len(X.shape) == 1:
        return wilcoxon(X, f).pvalue
    # now we shall perform and check each one...and return only the lowest pvalue
    return np.max([wilcoxon(x, f) for x in X.T])

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
        self.intragroup_alpha = intragroup_alpha
        self.intergroup_thres = intergroup_thres if intergroup_thres is not None else epsilon
    
    def _dpp_estimate_k(self, L):
        """
        L is the input kernel
        """
        """
        pca = PCA(n_components=None)
        pca.fit(L)
        pca_k = np.min(np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 
                                  (1-self.intragroup_alpha)))
        
        # also use KernelPCA
        kpca = KernelPCA(kernel='rbf')
        kpca.fit(L)
        kpca_k = np.argwhere(kpca.lambdas_ > 0.01).flatten().shape[0]
        self.dpp_k['pca'] = pca_k
        self.dpp_k['kpca'] = kpca_k
        """
        self.dpp_k['pca'] = None
        
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
        print(X.shape)
        cols_to_index = [idx for idx, x in enumerate(X_.columns) if x in self.coef_info['cols']]
        unseen_cols_to_index = [idx for idx, x in enumerate(X_.columns) if x not in self.coef_info['cols']]
        if X.shape[0] < 1000 or X.shape[1] < 100:
            feat_dist = rbf_kernel(X.T)
        else:
            feat_dist = Nystroem().fit_transform(X.T)
            feat_dist = feat_dist.dot(feat_dist.T)
        #self._dpp_estimate_k(feat_dist)
        #k = self.dpp_k['pca'] #- len(self.coef_info['cols'])
        k = None
        
        feat_index = []
        #while len(feat_index) == 0:
        if len(self.coef_info['cols']) == 0:
            feat_index = sample_dpp(decompose_kernel(feat_dist), k=k)
        else:            
            feat_index = sample_conditional_dpp(feat_dist, cols_to_index, k=k)
        feat_index = [x for x in feat_index if x is not None]
        
        # select features using entropy measure
        # how can we order features from most to least relevant first?
        # we chould do it using f test? Or otherwise - presume DPP selects best one first
        
        s_b, s_w = class_separability(X, y)
        col_sel = evaluate_feats(s_b, s_w)
        #sel_cols = list(self.coef_info['cols']) + list(col_sel)
        
        """
        feat_entropy = []
        excl_entropy = []
        X_sel = X[:, feat_index]
        
        for idx, feat in enumerate(X_sel.T):
            if len(feat_entropy) == 0:
                feat_entropy.append(idx)
                continue
            if entropy(X_sel[:, feat_entropy]) > entropy(X_sel[:, feat_entropy+[idx]]):
                feat_entropy.append(idx)
            else:
                excl_entropy.append(idx)
        """
        # iterate over feat_index to determine 
        # information on wilcoxon test
        # as the feat index are already "ordered" as that is how DPP would
        # perform the sampling - we will do the single pass in the same
        # way it was approached in the OGFS
        # feat index will have all previous sampled columns as well...
        if not self.unseen_only and len(feat_index) > 0:
            feat_check = []
            excl_check = []
            X_sel = X[:, feat_index]
            
            for idx, feat in enumerate(X_sel.T):
                if len(feat_check) == 0:
                    feat_check.append(idx)
                    continue
                wilcoxon_pval = wilcoxon_group(X_sel[:, feat_check], feat)
                #print("\tWilcoxon: {}".format(wilcoxon_pval))
                if wilcoxon_pval < self.intragroup_alpha:
                    feat_check.append(idx)
                else:
                    excl_check.append(idx)
            feat_check_ = (feat_check+col_sel)
            index_to_col = [col for idx, col in enumerate(X_.columns) if idx in feat_check_]
        elif self.unseen_only:
            # if we are considering unseen only, we will simply let the regulariser
            # act on it, sim. to grafting.
            index_to_col = [col for idx, col in enumerate(X_.columns) if idx in feat_index]
        else:
            # only use supervised criteria
            feat_check_ = (feat_check+col_sel)
            index_to_col = [col for idx, col in enumerate(X_.columns) if idx in feat_index]
        self.unseen_only = False # perhaps add more conditions around unseen - i.e. once unseen condition kicks in, it remains active?
        self.coef_info['cols'] = list(set(self.coef_info['cols'] + index_to_col))
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