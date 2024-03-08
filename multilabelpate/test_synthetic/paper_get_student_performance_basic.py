import pickle
import os
import random
import numpy as np
from multilabelpate.Aggregation import *
from skmultilearn.adapt import MLkNN
from sklearn.metrics import f1_score, recall_score, precision_score



def get_dp(rdps_sum, orders, delta):
    delta_term = np.divide(np.log(1.0 / delta), orders - 1)
    return np.min(rdps_sum + delta_term)




with open(os.path.join("data","all_preds_for_student.p"), "rb") as f:
    all_preds_for_student = pickle.load(f)

#parameter configuration
T=15
Ns_kept = [500,1000,2000]
noises = [3200,4800,6400]
orders = np.array([1.000001,2,4,8,16,32])
seeds = [14142,41024,5096,1014,124095,12495,8952,50912,4095,8252]
N=300
results = {}


for i in range(5):
    #Get data set
    with open(os.path.join("data","data_student",str(N)+"_"+str(i)), "rb") as f:
        dataset = pickle.load(f)
    X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]
    delta = 1.0 / (10**6)
    preds = all_preds_for_student[str(N)+","+str(i)]#preds for X_train
    res_dataset = {}
    for noise in noises:
        res_noise = {}
        for N_kept in Ns_kept:
            #Compute over different random subsets and take median later
            f1s_adapt = np.zeros(len(seeds))
            pres_adapt = np.zeros(len(seeds))
            recs_adapt = np.zeros(len(seeds))
            dps_adapt = np.zeros(len(seeds))
            f1s_fixed = np.zeros(len(seeds))
            pres_fixed = np.zeros(len(seeds))
            recs_fixed = np.zeros(len(seeds))
            dps_fixed = np.zeros(len(seeds))
            for k in range(len(seeds)):
                seed = seeds[k]
                random.seed(seed)
                #Take random subset of data:
                indices_subset = np.array(random.sample(list(range(len(y_train))), N_kept), dtype=int)
                preds_subset = preds[:,indices_subset,:]
                X_train_subset = X_train[indices_subset,:]

                #ADAPTIVE
                threshold_ensemble="adaptive"
                #get aggregated predictions
                aggregated_preds, rdps, _ = aggregate_gaussian(preds_subset, threshold_ensemble, noise,orders, T=T, seed=seed,ignore_privacy=False)
                rdp = np.sum(rdps, axis=0)
                #train student on predictions
                knn = MLkNN(k=10, s=1.0)
                knn.fit(X_train_subset, aggregated_preds)
                y_pred = knn.predict(X_test)
                f1s_adapt[k] = f1_score(y_test, y_pred, average="macro", zero_division=1.0)
                pres_adapt[k] = precision_score(y_test, y_pred, average="macro", zero_division=0.0)
                recs_adapt[k] = recall_score(y_test, y_pred, average="macro", zero_division=1.0)
                dps_adapt[k] = get_dp(rdp, orders, delta)
                #FIXED
                threshold_ensemble=0.3
                #Get aggregated predictions
                aggregated_preds, rdps, _ = aggregate_gaussian(preds_subset, threshold_ensemble*N, noise, orders, T=T, seed=seed,ignore_privacy=False)
                rdp = np.sum(rdps, axis=0) 
                #train student on predictions
                knn = MLkNN(k=10, s=1.0)
                knn.fit(X_train_subset, aggregated_preds)
                y_pred = knn.predict(X_test)
                f1s_fixed[k] = f1_score(y_test, y_pred, average="macro", zero_division=1.0)
                pres_fixed[k] = precision_score(y_test, y_pred, average="macro", zero_division=0.0)
                recs_fixed[k] = recall_score(y_test, y_pred, average="macro", zero_division=1.0)
                dps_fixed[k] = get_dp(rdp, orders, delta)

            #Get median dataset index according to f1*dp
            f1dp_adapt = np.multiply(f1s_adapt,dps_adapt)
            f1dp_fixed = np.multiply(f1s_fixed,dps_fixed)
            idx_adapt = np.argsort(f1dp_adapt)[len(f1dp_adapt)//2]
            idx_fixed = np.argsort(f1dp_fixed)[len(f1dp_fixed)//2]

            res_noise["N_kept="+str(N_kept)] = {
                "results_adapt": [pres_adapt[idx_adapt],recs_adapt[idx_adapt],f1s_adapt[idx_adapt], dps_adapt[idx_adapt]],
                "results_fixed": [pres_fixed[idx_fixed],recs_fixed[idx_fixed],f1s_fixed[idx_fixed], dps_fixed[idx_fixed]]
            }
        res_dataset["noise="+str(noise)] = res_noise
    results[str(N)+"_"+str(i)] = res_dataset
pickle.dump(results,open(os.path.join("results/","paper_get_student_performance_random_all_noises"),"wb"))

    
