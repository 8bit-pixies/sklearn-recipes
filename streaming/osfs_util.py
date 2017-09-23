from scipy.stats import chi2_contingency
from scipy import stats
import scipy
import numpy as np
import random
from scipy.stats import norm

def partial_cor(X, y):
    """
    Calculate all columns of X conditional on y
    returns a partial correlation matrix
    """
    cor_m = np.ones((X.shape[1], X.shape[1]))
    y_reshape = y.reshape(-1, 1)
    tri_idx = np.triu_indices(X.shape[1], 1)
    for i, j in zip(tri_idx[0].flatten(), tri_idx[1].flatten()):
        beta_i = scipy.linalg.lstsq(X[:, i].reshape(-1, 1), y_reshape)[0]
        beta_j = scipy.linalg.lstsq(X[:, j].reshape(-1, 1), y_reshape)[0]
        res_j = (X[:, j] - (y*beta_i)).flatten()
        res_i = (X[:, i] - (y*beta_j)).flatten()
        corr = stats.pearsonr(res_i, res_j)[0]
        cor_m[i, j] = corr
        cor_m[j, i] = corr
    return cor_m

# compute fisher info
def fisher_test(X, y):
    """
    Claculate score between x, y, and z
    """
    cor_m = partial_cor(X, y)
    #cor_m[cor_m == 0] = np.finfo(float).eps
    #cor_v = cor_m[0, 1]
    z_score = 0.5*np.log((1+cor_m)/(1-cor_m))
    z_n = len(set(list(y)))
    N = X.shape[0]
    test_stat = np.sqrt(N - z_n -3) * np.abs(z_score)
    p_val = 1-scipy.stats.norm.cdf(test_stat)
    np.fill_diagonal(p_val, float("inf"))
    return test_stat, p_val

# calculate weak and strong dependences
def dependence_test(X, y):
    teststat, pval = fisher_test(X, y)
    strong_dep = np.max(pval[np.triu_indices(pval.shape[0], 1)])
    weak_dep   = np.min(pval[np.triu_indices(pval.shape[0], 1)])
    return {
        'strong_dependence': strong_dep, 
        'weak_dependence': weak_dep
    }