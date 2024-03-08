import pickle
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from multilabelpate.Aggregation import aggregate_gaussian
from multilabelpate.Privacy import compute_q_confidence_all, rdp_gaussian_all

'''
Analysis of the confidence threshold: For several confidence tresholds, report: 
1.) Percentage of points filtered/crossing the check
2.) Performance of remaining data points
3.) DP Privacy Cost per data point crossed 
Do that over different choices of noises (for check and aggregation) and values of the parameter n

'''
def get_dp(rdps_sum, orders, delta):
    delta_term = np.divide(np.log(1.0 / delta), orders - 1)
    sum = rdps_sum+delta_term
    idx = np.argmin(sum)
    return sum[idx], idx

#PARAMS
T=15
fixed_threshold=0.3
Ns = [300]#probably gonna have dropped less teachers
orders = np.array([1.000001,2,4,8,16,32])#irrelevant here
seeds = [1234,1495,75329,5982,5149]
delta  = 1.0 / (10**6)
conf_breakpoints = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3]
noises = [[1600,200],[4800,3200], [9600,6400]]
ns = [1,5,10]

#get teacher predictions (=labels)
with open(os.path.join("data","all_predictions.p"), "rb") as f:
    all_preds_synthetic = pickle.load(f)


results = {} 
for i in range(len(Ns)):
    N = Ns[i]
    for j in range(5):
        seed = seeds[j]
        #Get data
        with open(os.path.join("data","data_teachers",str(N)+"_"+str(j)), "rb") as f:
            dataset = pickle.load(f)
        y_test = dataset["y_test"]
        preds = all_preds_synthetic[str(N)+"_"+str(j)]
        n_classes = y_test.shape[1]
        res_dataset = {}
        for noise_c, noise in noises:
            res_noise = {}
            for n in ns:
                res_n = {}
                rdps_total_adapt = np.zeros((len(conf_breakpoints)-1, len(orders)))
                rdps_total_fixed = np.zeros((len(conf_breakpoints)-1, len(orders)))
                performances_adapt = np.zeros((len(conf_breakpoints)-1, 3))
                performances_fixed = np.zeros((len(conf_breakpoints)-1, 3))
                pers_crossed = np.zeros(len(conf_breakpoints)-1)

                #Get aggregated datapoints
                predictions_binary = preds >= 0.5
                predictions_raw = np.sum(predictions_binary, axis=0)
                predictions_raw_top_n = np.partition(predictions_raw, -n, axis=1)[:,-n:]
                noise_scale_std = np.sqrt(noise_c)
                noise_ensemble_th_top_n = np.random.default_rng(seed=seed).normal(loc=0.0, scale=noise_scale_std,
                                                                size=predictions_raw_top_n.shape)
                predictions_raw_top_n_noised = predictions_raw_top_n + noise_ensemble_th_top_n
                for k in range(len(conf_breakpoints)-1):
                    #filter those crossing the confthr
                    conf_thr_low = conf_breakpoints[k]*N
                    conf_thr_low_exp = np.ones((predictions_raw.shape[0], n))*conf_thr_low
                    predictions_teacher_top_n_low = np.where(predictions_raw_top_n_noised > conf_thr_low_exp , 1.0, 0.0)
                    predictions_crossed_low = np.sum(predictions_teacher_top_n_low,axis=1)
                    indices_crossed = (predictions_crossed_low==n)
                    filtered_predictions = preds[:,indices_crossed]

                    #compute privacy cost of the confidence check
                    qs_confidence = compute_q_confidence_all(predictions_raw_top_n, noise_scale_std, conf_thr_low_exp)
                    rdps_check, _ = rdp_gaussian_all(qs_confidence,noise_scale_std,orders,1.0)
                    rdps_total_adapt[k,:] +=np.sum(rdps_check, axis=0)
                    rdps_total_fixed[k,:] +=np.sum(rdps_check, axis=0)

                    pers_crossed[k] = np.sum (indices_crossed) / preds.shape[1]
                    #Get Performance on remaining datapoints
                    if pers_crossed[k]>0:
                        #for adaptive
                        aggregated_preds, rdps_adapt, _ = aggregate_gaussian(filtered_predictions, "adaptive", noise,orders, T=T,seed=seed,ignore_privacy=False)#T=n_classes means no t-approx
                        performances_adapt[k,2] = f1_score(y_test[indices_crossed], aggregated_preds, average="macro", zero_division=1.0)
                        performances_adapt[k,0] = precision_score(y_test[indices_crossed], aggregated_preds, average="macro", zero_division=0.0)
                        performances_adapt[k,1] = recall_score(y_test[indices_crossed], aggregated_preds, average="macro", zero_division=1.0)
                        
                        #for fixed 
                        aggregated_preds_fixed, rdps_fixed, _ = aggregate_gaussian(filtered_predictions, fixed_threshold*N, noise, orders, T=T,seed=seed,ignore_privacy=False)
                        performances_fixed[k,2] = f1_score(y_test[indices_crossed], aggregated_preds, average="macro", zero_division=1.0)
                        performances_fixed[k,0] = precision_score(y_test[indices_crossed], aggregated_preds, average="macro", zero_division=0.0)
                        performances_fixed[k,1] = recall_score(y_test[indices_crossed], aggregated_preds, average="macro", zero_division=1.0)

                        rdps_total_adapt[k,:] += np.sum(rdps_adapt, axis=0)
                        rdps_total_fixed[k,:] += np.sum(rdps_fixed, axis=0)
                    else:
                        performances_adapt[k,:] = np.array([0,0,0])
                        performances_fixed[k,:] = np.array([0,0,0])
                    dp_cost_adapt, _ = get_dp(rdps_total_adapt[k,:], orders, delta) 
                    dp_cost_fixed, _ = get_dp(rdps_total_fixed[k,:], orders, delta) 
                    res_n[str(conf_breakpoints[k])+"_"+str(conf_breakpoints[k+1])] = {"per points": pers_crossed[k],
                                                                                      "performance adapt": performances_adapt[k,:],
                                                                                      "dp cost adapt": dp_cost_adapt,
                                                                                      "performance fixed": performances_fixed[k,:],
                                                                                      "dp cost fixed": dp_cost_fixed}
                res_noise["n="+str(n)] = res_n
            res_dataset["noise_c="+str(noise_c)+"_noise="+str(noise)] = res_noise
        results[str(N)+"_"+str(j)] = res_dataset
pickle.dump(results,open(os.path.join("results/","paper_analyseConfidenceCheck_thresholdBased"),"wb"))
        
            
            
        