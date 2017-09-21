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
    return np.min([wilcoxon(x, f) for x in X.T])

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
        print("PCA K: {}".format(self.dpp_k))
        print("L dim: {}".format(L.shape))
        
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
        if X.shape[0] < 1000:
            feat_dist = rbf_kernel(X.T)
        else:
            feat_dist = Nystroem().fit_transform(X.T)
        self._dpp_estimate_k(feat_dist)
        k = self.dpp_k['pca'] - len(self.coef_info['cols'])
        if k < 1:
            # this means k is possibly negative, reevaluate k based only on new incoming feats!
            self.unseen_only = True
            unseen_kernel = feat_dist[unseen_cols_to_index, :][:, unseen_cols_to_index]
            #k = max(self._dpp_estimate_k(unseen_kernel), int(unseen_kernel.shape[0] * 0.5)+1)            
            k = unseen_kernel.shape[0]
            print("Unseen only")
            print(k)
        if len(self.coef_info['cols']) == 0:
            feat_index = sample_dpp(decompose_kernel(feat_dist), k=k)
        else:            
            feat_index = sample_conditional_dpp(feat_dist, cols_to_index, k=k)
        
        # select features using entropy measure
        # how can we order features from most to least relevant first?
        # we chould do it using f test? Or otherwise - presume DPP selects best one first
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
        if not self.unseen_only:
            feat_check = []
            excl_check = []
            X_sel = X[:, feat_index]
            
            for idx, feat in enumerate(X_sel.T):
                if len(feat_check) == 0:
                    feat_check.append(idx)
                    continue
                if wilcoxon_group(X_sel[:, feat_check], feat) >= self.intragroup_alpha:
                    feat_check.append(idx)
                else:
                    excl_check.append(idx)
            index_to_col = [col for idx, col in enumerate(X_.columns) if idx in feat_check]
        else:
            # if we are considering unseen only, we will simply let the regulariser
            # act on it, sim. to grafting.
            index_to_col = [col for idx, col in enumerate(X_.columns) if idx in feat_index]
        self.unseen_only = False # perhaps add more conditions around unseen - i.e. once unseen condition kicks in, it remains active?
        self.coef_info['cols'] = list(set(self.coef_info['cols'] + index_to_col))
        col_rem = X_.columns.difference(self.coef_info['cols'])
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
        self.seen_cols = list(set(self.seen_cols + X.columns.tolist()))
        X = X[X.columns.difference(self.coef_info['excluded_cols'])]
        
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