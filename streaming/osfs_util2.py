from scipy.stats import chi2_contingency
from scipy import stats
import scipy
import numpy as np
import random
from scipy.stats import norm
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, pairwise_kernels
from sklearn.kernel_approximation import Nystroem
from sklearn import preprocessing

from itertools import combinations, permutations, chain

def fast_cor(X_all):
    # calculate correlation
    X_scaled = preprocessing.scale(X_all)
    if X_all.shape[1] < 0:
        K = pairwise_kernels(X_scaled.T, metric='cosine')
        return K
    else:
        K = Nystroem('cosine').fit_transform(X_scaled.T)
        if K.shape[0] != K.shape[1]:
            # we will have to use pairwise kernels...
            K = pairwise_kernels(X_scaled.T, metric='cosine')
            return K
        d_inv = np.sqrt(np.diag(np.diag(K)))
        corr = d_inv.dot(K).dot(d_inv)
        return corr

def fisher_test(cor_m, N, z_n):
    cor_m = np.minimum(cor_m, 0.9999)
    cor_m = np.maximum(cor_m, -0.9999)
    z_score = 0.5*np.log((1+cor_m)/(1-cor_m))
    test_stat = np.sqrt(N - z_n -3) * np.abs(z_score)
    p_val = 1-scipy.stats.norm.cdf(test_stat)
    return p_val

def powerset(iterable, max_size=3):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, max_size+1))

def partial_dep_test(x1, c, x_dat, x1_name, x_dat_names, max_search_depth=3, prev=[], early_stopping=False, alpha=0.05):
    max_search_depth = min(max_search_depth, len(x_dat_names))
    x_n = x_dat.shape[0]
    x_all = np.hstack(
        [x1.reshape(-1, 1), 
         c.reshape(-1, 1), 
         x_dat])
    x_cor = fast_cor(x_all)
    partial_cor = []
    for cond in list(powerset(range(2, x_cor.shape[1]), max_search_depth)):
        name_set = set([x_dat_names[x-2] for x in cond])
        check_prev_result = [x for x in prev if x['var'] == x1_name and x['cond'] == name_set]
        if len(
        check_prev_result
        ) > 0:
            partial_cor.append(check_prev_result[0])
            continue
        
        z_n = len(cond)
        idx = [0, 1] + list(cond)
        V = x_cor[idx, :][:, idx]
        V = np.nan_to_num(V)
        try:
            V_inv = np.linalg.pinv(V)
        except:
            V_inv = scipy.linalg.pinv(V)
        cor = -V_inv[0][1]/(np.sqrt(V_inv[0][0]*V_inv[1][1]))
        pval = fisher_test(cor, x_n, z_n)
        partial_cor.append({
            'var': x1_name,
            'cor': cor,
            'pval': pval,
            'cond': name_set
        })
        if early_stopping and pval > alpha:
            return partial_cor
    return partial_cor

