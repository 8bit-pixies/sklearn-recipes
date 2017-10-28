
import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from dpp import sample_dpp, decompose_kernel, sample_conditional_dpp

def dpp_sampler(X):
    """
    Takes in dataset and return 
    set of features based on:
    
    only dpp sampling...will extend for supervised/unsupervised
    criterion
    """
    feat_dist = rbf_kernel(X.T)
    feat_index = sample_dpp(decompose_kernel(feat_dist), k=None)
    return feat_index
