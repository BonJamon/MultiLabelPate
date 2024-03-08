import pickle
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from multilabelpate.Aggregation import aggregate_gaussian

'''
For different noise levels and both thresholds (adaptive, fixed), reports:
1.) Aggregation performances
2.) DP Privacy Costs
3.) Percentage of Times the (better) data dependent privacy bound is used


'''


def get_dp(rdps, orders, delta):
    rdps_sum = np.sum(rdps, axis=0)
    delta_term = np.divide(np.log(1.0 / delta), orders - 1)
    sum = rdps_sum+delta_term
    idx = np.argmin(sum)
    return sum[idx], idx

#Parameters
T=15
fixed_threshold=0.3
noise_levels = [50,200,400,800,1600,3200,4800,6400]
Ns = [50,100,300]
orders = np.array([1.000001,2,4,8,16,32])
deltas  = [1.0 / (10**5),1.0 / (10**6),1.0 / (10**6)] #equals order of amount of datapoints
seeds = [1234,1495,75329,5982,5149]

#get teacher predictions (=labels)
with open(os.path.join("data","all_predictions.p"), "rb") as f:
    all_preds_synthetic = pickle.load(f)


results = {} 
for i in range(len(Ns)):
    N = Ns[i]
    delta = deltas[i]
    for j in range(5):
        seed = seeds[j]
        #Get data
        with open(os.path.join("data","data_teachers",str(N)+"_"+str(j)), "rb") as f:
            dataset = pickle.load(f)
        y_test = dataset["y_test"]
        preds = all_preds_synthetic[str(N)+"_"+str(j)]
        n_classes = y_test.shape[1]
        res_dataset = {}
        for noise in noise_levels:
            #for adaptive
            aggregated_preds, rdps, per_dep = aggregate_gaussian(preds, "adaptive", noise, orders, T=T,seed=seed,ignore_privacy=False)#T=n_classes means no t-approx
            aggr_f1 = f1_score(y_test, aggregated_preds, average="macro", zero_division=1.0)
            aggr_pre = precision_score(y_test, aggregated_preds, average="macro", zero_division=0.0)
            aggr_rec = recall_score(y_test, aggregated_preds, average="macro", zero_division=1.0)
            dp, idx = get_dp(rdps, orders, delta)
            per_dep = per_dep[idx]
            
            #for fixed 1 
            aggregated_preds_fixed, rdps_fixed, per_dep_fixed = aggregate_gaussian(preds, fixed_threshold*N, noise, orders, T=T,seed=seed,ignore_privacy=False)
            aggr_f1_fixed = f1_score(y_test, aggregated_preds, average="macro", zero_division=1.0)
            aggr_pre_fixed = precision_score(y_test, aggregated_preds, average="macro", zero_division=0.0)
            aggr_rec_fixed = recall_score(y_test, aggregated_preds, average="macro", zero_division=1.0)
            dp_fixed, idx = get_dp(rdps_fixed, orders, delta)
            per_dep_fixed = per_dep_fixed[idx]
            
            
            res_dataset[str(noise)] = {"results adapt": [aggr_pre, aggr_rec, aggr_f1, dp, per_dep],
                                       "results fixed": [aggr_pre_fixed, aggr_rec_fixed, aggr_f1_fixed, dp_fixed, per_dep_fixed]}
            
        results[str(N)+"_"+str(j)] = res_dataset
pickle.dump(results,open(os.path.join("results/","paper_evaluatePrivacy"),"wb"))
        
            
            
        