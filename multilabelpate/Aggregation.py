import numpy as np
from multilabelpate.Privacy import *
from multilabelpate.t_approx import t_approximation_alt



def calc_adaptive_threshold(predictions):
    '''
    Calculates threshhold based on biggest difference in histogram of teacher votes
    :param predictions: summed teacher votes
    :return: 1-dim numpy array: Threshhold for each datapoint
    '''
    ordered_preds = np.array(predictions)
    ordered_preds.sort(axis=1)#sort outputs for classes
    distances = ordered_preds[:,1:] - ordered_preds[:,:-1] #Remark: um 1 verschoben zu ordered_preds!
    max_dist_idx = np.argmax(distances, axis=1) #Want argmax over classes --> returns 1-dim array: for each datapoint the index
    
    res = [ordered_preds[i,max_dist_idx[i]] + distances[i,max_dist_idx[i]] / 2 for i in range(len(max_dist_idx))]
    return np.array(res)


def aggregate_gaussian(predictions,
                    threshold_ensemble,
                    noise_scale_ensemble,
                    orders,
                    T,
                    seed,
                    ignore_privacy = False,
                    debug=False):
    """
    basic aggregation (GNThreshold) for multiple data points simultaneously by
    1. Use t-approximation (for no t-approx use T=n_classes)
    2. sum all teacher votes
    3. compute privacy costs
    4. check summed teacher ensemble predictions with noise against thresholds
    :param noise_scale_ensemble: std of noise applied to the sum of the teacher predictions (should be between 0 and 1)
    :param threshold_ensemble: threshold the noised teacher votes have to cross to have a positive label
    :param predictions: numpy array with predictions of teachers
    :param debug:
    :param T: For t_approx
    :param orders: values of lambda for rdp 
    :param ignore_privacy: Useful not to use noise for some tests of the method
    :param seed: Seed used to generate noise
    :return:
    """

    n_teachers = predictions.shape[0]
    n_classes = predictions.shape[2]

    #1.) T-approximation
    preds_binary = t_approximation_alt(predictions, T)

    #2.) Sum all teacher votes
    predictions_raw = np.sum(preds_binary, axis=0)
    
    #Get threshold of shape n_datapoints, n_classes
    if type(threshold_ensemble)==str and threshold_ensemble == 'adaptive': #Das geht? Is ja sonst ne Zahl
        threshold_ensemble = calc_adaptive_threshold(predictions=predictions_raw)
        threshold_ensemble = np.expand_dims(threshold_ensemble, axis=1)
        threshold_expanded = threshold_ensemble
        for i in range(n_classes-1):
            threshold_expanded = np.hstack((threshold_expanded,threshold_ensemble))
    else:
        #Have fixed float value
        threshold_expanded = np.ones((predictions_raw.shape[0], predictions_raw.shape[1]))*threshold_ensemble
 
    
    #Generate noise
    noise_scale_std = np.sqrt(noise_scale_ensemble)
    noise = np.random.default_rng(seed=seed).normal(loc=0.0,
                                            scale=noise_scale_std,
                                            size=predictions_raw.shape)
    
    if not ignore_privacy:
        #3.) Compute rdp cost of shape (n_datapoints,n_orders)
        qs = compute_q_gaussian_all(predictions_raw, noise_scale_std, threshold_expanded)
        rdps, per_dep = rdp_gaussian_all(qs, noise_scale_std, orders, sensitivity=T)
        #4.) Get aggregated labels
        result = np.where(predictions_raw + noise > threshold_expanded, 1.0, 0.0) #threshhold for each datapoint
        return result, rdps, per_dep
    else:
        return np.where(predictions_raw > threshold_expanded, 1.0, 0.0)




def aggregate_gaussian_confidence(predictions,
                                threshold_ensemble_confidence,
                                noise_scale_ensemble_confidence,
                                noise_scale_ensemble,
                                threshold_ensemble,
                                orders,
                                T,
                                n,
                                seed,
                                ignore_privacy = False,
                                debug=False):
    """
    confident aggregation (Confident GNThreshold) for multiple data points simultaneously by
    1.) sum all teacher votes
    2.) get noised top-n predictions
    3.) get privacy costs of the confidence check
    4.) filter data points
    5.) use basic aggregation on filtered data points
    :param threshold_ensemble_confidence: threshold the top-n teacher predictions all have to cross
    :param threshold_ensemble: threshold for the basic aggregation
    :param noise_scale_ensemble: std of noise applied to the basic aggregation
    :param noise_scale_ensemble_confidence: std of noise applied to the sum of the teacher predictions (should be between 0 and 1)
    :param predictions: numpy array with predictions of teachers
    :param T: value of T for t-approximation of the basic aggregation
    :param n: count of top-n summed teacher predictions looked at
    :param orders: values of lambda for rdp 
    :param ignore_privacy: Useful not to use noise for some tests of the method
    :param seed: Seed used to generate noise
    :param debug:
    :return: aggregated labels of filtered datapoints, indices of filtered datapoints in original data, rdp costs confidence check, rdp cost basic aggregation, percentage of data dependent bound used for privacy cost in the confidence check, percentage of data dependent bound used for privacy cost in the aggregation
    """
    n_teachers = predictions.shape[0]
    n_classes = predictions.shape[2]
    
    #1.) sum all teacher votes
    predictions_binary = predictions >= 0.5
    predictions_raw = np.sum(predictions_binary, axis=0)
    predictions_raw_top_n = np.partition(predictions_raw, -n, axis=1)[:,-n:]

    #Get threshold of shape (n_datapoints,n)
    if type(threshold_ensemble_confidence)==str and threshold_ensemble_confidence == 'adaptive': #Das geht? Is ja sonst ne Zahl
        threshold_ensemble_confidence = calc_adaptive_threshold(predictions=predictions_raw)
        threshold_ensemble_confidence = np.expand_dims(threshold_ensemble_confidence, axis=1)
        threshold_expanded = threshold_ensemble_confidence
        for i in range(n-1):
            threshold_expanded = np.hstack((threshold_expanded,threshold_ensemble_confidence))
    else:
        threshold_expanded = np.ones((predictions_raw.shape[0], n))*threshold_ensemble_confidence

    #2.) Get noised top_n predictions
    noise_scale_std = np.sqrt(noise_scale_ensemble_confidence)
    noise_ensemble_th_top_n = np.random.default_rng(seed=seed).normal(loc=0.0, scale=noise_scale_std,
                                                    size=predictions_raw_top_n.shape)
    predictions_teacher_top_n = np.where(predictions_raw_top_n + noise_ensemble_th_top_n > threshold_expanded , 1.0, 0.0)

    #3.) Get Privacy costs
    if not ignore_privacy:
        #Only have Boolean decision: Keep Datapoint or Not --> Sensitivity = 1
        qs_confidence = compute_q_confidence_all(predictions_raw_top_n, noise_scale_std, threshold_expanded)
        rdps_confidence, pers_dep_c = rdp_gaussian_all(qs_confidence,noise_scale_std,orders,1.0)
    else:
        rdps_confidence = 0

    #4.) Filter predictions
    predictions_crossed = np.sum(predictions_teacher_top_n,axis=1) #if all top n 1 (above threshold), then crossed
    indices_crossed = (predictions_crossed == n) 
    filtered_predictions = predictions[:,indices_crossed] 

    #5.) use basic aggregation on remaining predictions
    if not ignore_privacy:
        labels, rdps, pers_dep = aggregate_gaussian(predictions=filtered_predictions,
                                threshold_ensemble=threshold_ensemble,
                                noise_scale_ensemble=noise_scale_ensemble,
                                orders=orders,
                                T=T,
                                seed=seed)
        
        return labels, indices_crossed, rdps_confidence, rdps, pers_dep_c, pers_dep
    else:
        labels = aggregate_gaussian(predictions=filtered_predictions,
                                threshold_ensemble=threshold_ensemble,
                                noise_scale_ensemble=noise_scale_ensemble,
                                orders=orders,
                                T=T,
                                seed=seed,
                                ignore_privacy=ignore_privacy)
        return labels, indices_crossed
    



def get_confident_predictions(predictions,threshold):
    '''
    Gets predictions of shape [datapoints,classes] and checks if any class for a datapoint crossed the threshhold
    :return: indices of crossed datapoints of shape (datapoints,)
    '''
    predictions_binary = np.where(predictions > threshold, 1.0, 0.0)
    predictions_crossed = np.sum(predictions_binary,axis=1)
    indices_crossed = predictions_crossed>0
    return predictions_binary[indices_crossed], indices_crossed


def aggregate_gaussian_interactive(predictions, 
                                   prediction_student,
                                   threshold_ensemble,
                                   threshold_ensemble_confidence,
                                   threshold_ensemble_interactive,
                                   threshold_student_confidence,
                                   noise_scale_ensemble_confidence,
                                   noise_scale_ensemble,
                                   T,
                                   n,
                                   orders,
                                   seed,
                                   ignore_privacy = False,
                                   debug=False):
    '''
    Interactive aggregation (Interactive GNThreshold) for single data point by
    1.) summing all teacher votes
    2.) computing privacy cost of agreement check
    3.) checking agreement
    4.1) using basic aggregation
    4.2) student confidence check 
    :param threshold_ensemble_confidence: threshold for the aggreement check
    :param threshold_ensemble: threshold for the basic aggregation
    :param threshold_ensemble_interactive: threshold for the student prediction to reinforce
    :param noise_scale_ensemble: std of noise applied to the basic aggregation
    :param noise_scale_ensemble_confidence: std of noise applied to the sum of the teacher predictions (should be between 0 and 1)
    :param predictions: numpy array with predictions of teachers
    :param prediction_student: student prediction
    :param T: value of T for t-approximation of the basic aggregation
    :param n: count of top-n summed teacher predictions looked at
    :param orders: values of lambda for rdp 
    :param ignore_privacy: Useful not to use noise for some tests of the method
    :param seed: Seed used to generate noise
    :param debug:
    :return: None if datapoint not chosen, labels of shape (n_classes) if point is chosen, rdp cost of interactive check, rdp cost of aggregation, boolean whether reinforced
    '''
    n_teachers = predictions.shape[0]

    #1.) sum all teacher votes
    predictions_binary = predictions >= 0.5
    prediction_student_binary = prediction_student > threshold_student_confidence
    prediction_raw = np.sum(predictions_binary, axis=0)
    term_check_raw = prediction_raw - n_teachers * prediction_student
    term_check_raw_top_n = np.partition(term_check_raw, -n, axis=0)[-n:]
    reinforced = False
    
    if not ignore_privacy:
        noise_ensemble_th = np.random.default_rng(seed=seed).normal(loc=0.0, scale=np.sqrt(noise_scale_ensemble_confidence),
                                                            size=term_check_raw_top_n.shape)

        threshold_expanded = np.ones(n)*threshold_ensemble_confidence
        prediction_no_agreement = np.where(term_check_raw_top_n + noise_ensemble_th > threshold_expanded, 1.0, 0.0)
        
        #2.) compute privacy cost of agreement check
        q_interactive = compute_q_confidence(term_check_raw_top_n, np.sqrt(noise_scale_ensemble_confidence), threshold_expanded)
        rdp_interactive = rdp_gaussian(q_interactive, np.sqrt(noise_scale_ensemble_confidence), orders, 1.0)
        
        #3.) agreement check
        if np.sum(prediction_no_agreement)==n:
            #4.1) basic aggregation
            label, rdp, _ = aggregate_gaussian(predictions=np.expand_dims(predictions, axis=1), threshold_ensemble=threshold_ensemble,
                                    noise_scale_ensemble=noise_scale_ensemble, orders=orders, T=T, seed=seed)
            return label[0,:], rdp_interactive, rdp[0], reinforced
        else:
            #4.2) student confidence check
            if np.sum(prediction_student_binary) > threshold_ensemble_interactive:
                reinforced = True
                return prediction_student_binary, rdp_interactive, 0.0, reinforced
            else:
                return None, rdp_interactive, 0.0, reinforced
    else:
        threshold_expanded = np.ones(n)*threshold_ensemble_confidence
        prediction_no_agreement = np.where(term_check_raw_top_n > threshold_expanded, 1.0, 0.0)
        if np.sum(prediction_no_agreement)==n:
            #aggregate gaussian for shape [teacher,datapoint,classes], returns of shape [datapoint,classes]
            label = aggregate_gaussian(predictions=np.expand_dims(predictions, axis=1), threshold_ensemble=threshold_ensemble,
                                    noise_scale_ensemble=noise_scale_ensemble, orders=orders, T=T, seed=seed, ignore_privacy=True)
            return label[0,:], reinforced
        else:
            #return student prediction if agreement with teacher and student confident (measure as is predicting at least threshold_interactive positive labels)
            if np.sum(prediction_student_binary) > threshold_ensemble_interactive:
                reinforced = True
                return prediction_student>0.5, reinforced
            else:
                return None, reinforced
    
