import math
import numpy as np
import scipy.stats


##########
# Basics #
##########

def _log1mexp(x):
    """Numerically stable computation of log(1-exp(x))."""
    if x < -1:
        return math.log1p(-math.exp(x))
    elif x < 0:
        return math.log(-math.expm1(x))
    elif x == 0:
        return -np.inf
    else:
        raise ValueError("Argument must be non-positive.")
    
    
def _log1mexp_all(x):
    ret = x.copy()
    ret[x < -1] = np.log1p(-np.exp(x[x < -1]))
    ret[np.logical_and(x < 0, x >= -1)] = np.log(-np.expm1(x[np.logical_and(x < 0, x >= -1)]))
    ret[x == 0] = -np.inf
    if np.any(x > 0):
        raise ValueError("Arguments must be non-positive.")
    return ret


def compute_eps_from_delta(orders, rdp, delta):
    """Translates between RDP and (eps, delta)-DP.

  Args:
    orders: A list (or a scalar) of orders.
    rdp: A list of RDP guarantees (of the same length as orders).
    delta: Target delta.

  Returns:
    Pair of (eps, optimal_order).

  Raises:
    ValueError: If input is malformed.
  """
    if np.sum(rdp) == 0.0:
        return 0, 0 # if noise scale = 0.0
    if len(orders) != len(rdp):
        raise ValueError("Input lists must have the same length.")
    eps = np.array(rdp) - math.log(delta) / (np.array(orders) - 1)
    idx_opt = np.argmin(eps)
    return eps[idx_opt], orders[idx_opt]




def compute_q_gaussian_all(counts, sigma, thresholds):
    '''
    returns upper bounds on q for each datapoint
    :param thresholds: needs to be of same shape as counts
    :param counts: raw summed teacher votes in shape (n_datapoints, n_classes)
    :param sigma: Standard deviation of Gaussian noise
    :returns: array of shape #datapoints
    '''
    thresholds_delta = np.abs(counts-thresholds)
    cdf = scipy.stats.norm.cdf(thresholds_delta, scale=sigma, loc=0)
    qs = 1 - np.product(cdf,axis=1)
    
    amount_possible_outcomes = 2 ** counts.shape[1]
    random_outcome = 1 - 1 / amount_possible_outcomes
    random_outcomes = np.ones(counts.shape[0]) * random_outcome
    
    return np.minimum(qs,random_outcomes)
  
    
    

def compute_q_gaussian(counts, sigma, threshold):
    """
    Returns an upper bound on Pr[outcome != likely outcome with noise] for GNThreshold. --> this is the q

    :param counts: raw summed votes of all teachers for the classes (array with length of num_classes)
    :param sigma: Standard deviation of Gaussian noise
    :param threshold: ensemble threshold up from that the votes should be considered as 1/0
    :return: likelihood that not the regular result gets returned (noise turn result for classes)
    """
    # can take the abs as gaussian is symmetrical
    threshold_delta = np.abs(counts - threshold)

    # Upper bound q consists of the sum of all potential errors
    # As it can flip only if the noise goes into one direction max probability per class is 0.5
    # single sigma as it can flip exactly one vote
    # sf = scipy.stats.norm.sf(threshold_delta, scale=sigma, loc=0)
    cdf = scipy.stats.norm.cdf(threshold_delta, scale=sigma, loc=0)
    # q = np.sum(sf)
    q = 1 - np.product(cdf)

    # calculate random outcome based on equal probability for any outcome
    amount_possible_outcomes = 2 ** len(counts)
    random_outcome = 1 - 1 / amount_possible_outcomes

    return min(q, np.array(random_outcome))






def rdp_gaussian_all(qs, sigma, orders, sensitivity):
    """
    Compute rdp cost: takes minimum of data independent and data dependent gaussian for multiple data points simultaneously
    :param qs: 
    :param sigma: 
    :param orders:
    :param sensitivity:
    """
    #initialize 0 --> where there is q==0 or log_q == -inf here autoatically return 0 
    orders = np.atleast_1d(orders)
    n_orders = orders.shape[0]
    n_datapoints = qs.shape[0]
    
    log_qs = np.log(qs)
    if np.any(log_qs>0) or np.any(orders<1):
        raise ValueError("Inputs are malformed.")
    
    #1D mask for datapoints
    zero_mask =  ((qs == 0) | (np.isneginf(log_qs)))
    zero_mask_expanded = np.tile(np.expand_dims(zero_mask, axis=1), (1,n_orders))
    
    variance = sigma ** 2
    
    #mu1 and mu2 according to proposition 10 
    mu_hi2 = np.sqrt(2*variance* (-log_qs) / sensitivity) + 0.00000001
    mu_hi1 = mu_hi2 + 1.0
    
    
    #2D Values
    mu_hi1_expanded = np.tile(np.expand_dims(mu_hi1, axis=1), (1,n_orders))
    mu_hi2_expanded = mu_hi1_expanded - 1
    orders_expanded = np.tile(np.expand_dims(orders, axis=0), (n_datapoints,1))
    log_qs_expanded = np.tile(np.expand_dims(log_qs, axis=1), (1,n_orders))
    
    #Dont need mu2 and mu1 where there is q==0: Set to 0 to avoid numerical problems
    mu_hi1_expanded[zero_mask_expanded] = 0.0
    mu_hi2_expanded[zero_mask_expanded] = 0.0
    
    #Initialize data-independent bound
    ret = (orders_expanded * sensitivity) / (2 * variance)
    
    # data-independent bounds for estimation
    # corresponds to epsilon_1
    rdp_hi1_expanded = (mu_hi1_expanded * sensitivity) / (2 * variance)
    # corresponds to epsilon_2
    rdp_hi2 = (mu_hi2 * sensitivity) / (2 * variance)
    rdp_hi2_expanded = np.tile(np.expand_dims(rdp_hi2, axis=1), (1,n_orders))
    
    log_a2 = np.multiply(mu_hi2 - 1, rdp_hi2)
    
    #2D mask for deciding if data-dependent bound is applicable
    #only take data dependent bound for mu2>1
    compare_mu_hi2 = mu_hi2_expanded > 1
    mask1 = (mu_hi1_expanded > orders_expanded) & compare_mu_hi2
    #only take data dependent bound for q <= ... (from Theorem 6)
    #remark: Compute only where q not zero or log_q = -inf
    #Problem: What if 0 < mu_hi2 < 1? --> might take log of number < 1 --> q cannot be below that --> make sure it returns false
    mask2 = np.zeros(zero_mask.shape) < 0#force boolean
    maskmask = np.logical_and(np.invert(zero_mask), mu_hi2>=1.0)#filter out where we dont want to compute 
    mask2[maskmask] = (log_qs[maskmask] <= log_a2[maskmask] - np.multiply(mu_hi2[maskmask], np.log(1 + np.divide(1,mu_hi1[maskmask] - 1)) + np.log(1 + np.divide(1, mu_hi2[maskmask] - 1)))) 
    mask2 = np.tile(np.expand_dims(mask2, axis=1), (1,n_orders))
    mask3 = (-log_qs > rdp_hi2)
    mask3 = np.tile(np.expand_dims(mask3, axis=1), (1,n_orders))
    #only take data dependent bound where q has valid value (not zero or -inf)
    mask = mask1 & mask2 & mask3 & np.invert(zero_mask_expanded)
    
    condition = np.divide(np.exp(np.multiply(mu_hi2-1,rdp_hi2)),np.multiply(np.divide(mu_hi2+1,mu_hi2), np.divide(mu_hi2, mu_hi2-1)))
    #Compute only for masked cases
    log1q = _log1mexp_all(log_qs_expanded[mask])
    log_a = np.multiply(orders_expanded[mask] - 1, 
                        log1q - _log1mexp_all(np.multiply(log_qs_expanded[mask] + rdp_hi2_expanded[mask], 1 - np.divide(1, mu_hi2_expanded[mask]))))
    log_b = np.multiply(orders_expanded[mask] - 1, rdp_hi1_expanded[mask] - np.divide(log_qs_expanded[mask], (mu_hi1_expanded[mask] - 1)))
    log_s = np.logaddexp(log1q + log_a, log_qs_expanded[mask] + log_b)
    #initialize with data independent privacy bound
    res_dependend = np.divide(log_s, orders_expanded[mask] - 1)
    #where data dependent bound is applicable and better, take it
    ret[mask] = np.where(ret[mask] < res_dependend, ret[mask], res_dependend)

    #compute percentage of times data dependent bound is used
    def mean_per(x):
        max_el = np.max(x)
        is_eq = x==max_el
        return 1.0 - (np.sum(is_eq) / len(x))
    if ret.shape[0]!=0:
        pers_dep = np.apply_along_axis(mean_per, 0, ret)
    else:
        pers_dep= 0.0
    
    return ret, pers_dep
    

    
    
    
    

def rdp_gaussian(q, sigma, orders, sensitivity):
    """

    :param q:
    :param sigma:
    :param orders:
    :param sensitivity:
    :return:
    """
    """Bounds RDP from above of GNMax given an upper bound on q (Theorem 6).

  Args:
    logq: Natural logarithm of the probability of a non-argmax outcome.
    sigma: Standard deviation of Gaussian noise.
    orders: An array_like list of Renyi orders.

  Returns:
    Upper bound on RPD for all orders. A scalar if orders is a scalar.

  Raises:
    ValueError: If the input is malformed.
    :param q:
    :param n_classes:
  """
    # If the mechanism's output is fixed, it has 0-DP.
    if q == 0.:
        if np.isscalar(orders):
            return 0.
        else:
            return np.full_like(orders, 0., dtype=np.float16)

    # transfer to log scale
    logq = np.log(q)

    # not defined for alpha=1
    if logq > 0 or sigma < 0 or np.any(orders < 1):
        raise ValueError("Inputs are malformed.")

    # If the mechanism's output is fixed, it has 0-DP.
    if np.isneginf(logq):
        if np.isscalar(orders):
            return 0.
        else:
            return np.full_like(orders, 0., dtype=np.float16)

    variance = sigma ** 2

    # Use two different higher orders: mu_hi1 and mu_hi2 computed according to theorem in thesis
    # proposition 10 original paper
    mu_hi2 = math.sqrt(2 * variance * -logq / sensitivity)
    mu_hi1 = mu_hi2 + 1

    orders_vec = np.atleast_1d(orders)

    # baseline: data-independent bound
    # proposition 8 original paper
    ret = (orders_vec * sensitivity) / (2 * variance)

    # Filter out entries where data-dependent bound does not apply.
    # practical application original paper
    mask = np.logical_and(mu_hi1 > orders_vec, mu_hi2 > 1)

    # data-independent bounds for estimation
    # corresponds to epsilon_1
    rdp_hi1 = (mu_hi1 * sensitivity) / (2 * variance)
    # corresponds to epsilon_2
    rdp_hi2 = (mu_hi2 * sensitivity) / (2 * variance)

    # pre-condition
    # practical implementation original paper
    log_a2 = (mu_hi2 - 1) * rdp_hi2


    # Make sure q is in the increasing wrt q range and A is positive.
    if (np.any(mask) and logq <= log_a2 - mu_hi2 *
            (math.log(1 + 1 / (mu_hi1 - 1)) + math.log(1 + 1 / (mu_hi2 - 1))) and
            -logq > rdp_hi2):
        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1q = _log1mexp(logq)  # log1q = log(1-q)
        log_a = (orders - 1) * (
                log1q - _log1mexp((logq + rdp_hi2) * (1 - 1 / mu_hi2)))

        log_b = (orders - 1) * (rdp_hi1 - logq / (mu_hi1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s = np.logaddexp(log1q + log_a, logq + log_b)
        # get minimum of data dep and indep; take min if conditions are met
        ret[mask] = np.minimum(ret, log_s / (orders - 1))[mask]
    assert np.all(ret >= 0)

    if np.isscalar(orders):
        return np.ndarray.item(ret)
    else:
        return ret





def compute_pr_not_answered_all(counts, sigma, answer_threshold):
    '''
    Computes probability that noisy threshold is crossed. Assume same threshold and sigma for all datapoints
    :param counts: of shape #datapoints,#classes
    :returns: probabilities of noise changing whether crossed in #datapoints
    '''
    max_value = np.abs(counts - answer_threshold)
    above_th = np.where(counts > answer_threshold, 1.0, 0.0)
    chance_that_noise_flips = scipy.stats.norm.sf(max_value, scale=sigma)

    
    #index for datapoints where any class is above the threshold 
    idx_any = (np.sum(above_th, axis=1) > 0)
    #Multiply all probabilities of falling under threshold for each value above for each datapoint
    probs_any = np.multiply(above_th[idx_any,:], chance_that_noise_flips[idx_any,:])
    probs_any = np.where(probs_any > 0, probs_any, 1.0)
    probs_any = np.product(probs_any, axis=1)
     
    #index for datapoints where all points are above threshold
    idx_all = np.invert(idx_any)
    chance_that_noise_doesnt_flip = 1 - chance_that_noise_flips
    probs_all = 1 - np.product(chance_that_noise_doesnt_flip[idx_all,:], axis=1)
    
    probs = np.zeros(counts.shape[0])
    probs[idx_any] = probs_any
    probs[idx_all] = probs_all
    
    return probs

    


def compute_pr_not_answered(counts, sigma_threshold, answer_threshold):
    """
    Computes probability that a noisy threshold is crossed.
    It reports the prob of the most likely class (highest score) with applied noise.
    :param answer_threshold: number of teachers that must agree to answer the query at all
    :param sigma: Standard deviation of Gaussian noise.
    :param counts: summed votes of all teachers for the classes (array with length of num_classes)
    :return: probability that max is larger than a noisy threshold.
    """
    # the most likely outcome depends on
    # 1. values above the threshold --> how likely it is to keep them above the threshold (none of them can flip)
    # 2. values below the threshold --> how likely it is that any of them flip
    
    max_value = np.abs(counts - answer_threshold)
    above_th = np.where(counts > answer_threshold, 1.0, 0.0)
    chance_that_noise_flips = scipy.stats.norm.sf(max_value, scale=sigma_threshold)

    if np.sum(above_th) >= 1.0:
        # case 1.) any value above threshold
        # event of no answer can only occur if all values above the threshold flip
        # P(all of the above th flip) = P(all flip) = product of all events
        res = 1.0
        for idx in range(len(counts)):
            if above_th[idx] == 1.0:
                res *= chance_that_noise_flips[idx]
    else:
        # case 2.) no value above threshold
        # event that any of the values below the threshold flips
        # P(any of the below threshold flips) = 1-P(none of the below th ones flip)
        # = 1 - product for all of them to stay as they are
        res = 1.0
        for idx in range(len(counts)):
            if above_th[idx] == 0.0:
                res *= (1 - chance_that_noise_flips[idx])
        res = 1 - res
    return res

def compute_q_confidence(counts, sigma, threshold):
    '''
    """
    Returns an upper bound on Pr[outcome != likely outcome with noise] for confidence check of Confident GNThreshold --> this is the q

    :param counts: raw summed votes of all teachers for the classes (array with length of num_classes)
    :param sigma: Standard deviation of Gaussian noise
    :param threshold: ensemble threshold up from that the votes should be considered as 1/0
    :return: likelihood that not the regular result gets returned (noise turn result for classes)
    """
    '''
    n = counts.shape[0]
    diff = np.abs(counts-threshold)
    above_th = np.where(counts > threshold, 1.0,0.0)
    chance_that_noise_flips = scipy.stats.norm.sf(diff, scale=sigma)
    if np.sum(above_th)==n:
        #Case: confidence check is crossed. Need to compute prob that any value drops below threshold
        res = 1.0
        for i in range(n):
            res *= (1-chance_that_noise_flips[i])
        return 1.0 - res
    else:
        #Case: Confidence check is not crossed. Need to compute probability that all values above stay above and all below flip
        p_alt = np.where(above_th, 1-chance_that_noise_flips, chance_that_noise_flips)
        res = 1.0
        for i in range(n):
            res *= p_alt[i]
        return res
    
def compute_q_confidence_all(counts, sigma, threshold):
    '''
    Returns an upper bound on Pr[outcome != likely outcome with noise] for confidence check of Confident GNThreshold for multiple datapoints--> this is the q

    :param counts: raw summed votes of all teachers for the classes (array with length of num_classes)
    :param sigma: Standard deviation of Gaussian noise
    :param threshold: ensemble threshold up from that the votes should be considered as 1/0
    :return: likelihood that not the regular result gets returned (noise turn result for classes)
    '''
    n = counts.shape[1]
    diff = np.abs(counts-threshold)
    above_th = np.where(counts > threshold, 1.0,0.0)
    chance_that_noise_flips = scipy.stats.norm.sf(diff, scale=sigma)
    
    above_th_sum = np.sum(above_th, axis=1)
    idx_crossed = above_th_sum==n
    idx_crossed_exp = np.matmul(np.expand_dims(idx_crossed,axis=1), np.ones((1,n)))== 1.0
    probs = np.zeros(counts.shape[0])
    #Case1: confidence check crossed: Need to compute probability that any value flips
    probs[idx_crossed] = 1.0 - np.product(1.0 - chance_that_noise_flips[idx_crossed], axis=1)
    #Case2: confidence check not crossed: need to compute probability that all above stay above and all below flip
    probs_alt = np.where(idx_crossed_exp, 1.0 - chance_that_noise_flips, chance_that_noise_flips)
    probs[np.invert(idx_crossed)] = np.product(probs_alt[np.invert(idx_crossed)], axis=1)
    return probs


    
    
