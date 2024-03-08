import numpy as np

def t_approximation(preds, t):
    '''
    Takes binary predictions of shape (n_datapoints, n_teacher, n_classes) 
    and returns an array of the same shape, where each teacher predicts t classes at most
    '''
    count_votes = np.sum(preds, axis=2)+10e-10
    scales = np.minimum(np.divide(t, count_votes),np.ones(preds.shape[0:2]))
    scales_expanded = np.tile(np.expand_dims(scales, axis=2), (1,preds.shape[2]))
    approximation = np.multiply(preds, scales_expanded)
    return approximation

def t_approximation_alt(preds, t):
    '''
    Takes probabilistic predictions (in [0,1]) of shape (n_datapoints, n_teacher, n_classes) 
    and returns a binary array of the same shape, where each teacher predicts t classes at most
    '''
    idx_ordered = np.argsort(preds, axis=2)
    rank_idx = np.argsort(idx_ordered, axis=2)
    preds_binary = np.where((rank_idx >= preds.shape[2] - t) & (preds >= 0.5), 1.0, 0.0)
    return preds_binary





