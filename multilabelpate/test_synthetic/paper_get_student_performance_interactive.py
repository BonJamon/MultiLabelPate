import pickle
import os
import numpy as np
import random
from sklearn.metrics import f1_score, recall_score, precision_score
from multilabelpate.Aggregation import aggregate_gaussian, aggregate_gaussian_interactive
from multilabelpate.Privacy import compute_q_confidence_all, rdp_gaussian_all
from skmultilearn.adapt import MLkNN
from sklearn.utils import shuffle

'''
For fixed noise: Test over different confidenceThresholds: how many points aggregated by teacher + f1 + dp cost, how many points reinforced by student for studentConfidenceThreshold and nInteractive
Do that over 3 different student confidence thresholds and nInteractive --> 3x3 plots 

'''
def get_dp(rdps_sum, orders, delta):
    delta_term = np.divide(np.log(1.0 / delta), orders - 1)
    sum = rdps_sum+delta_term
    return np.min(sum)


#Parameters
T=15
fixed_threshold=0.3
N = 300
orders = np.array([1.000001,2,4,8,16,32])
delta  = 1.0 / (10**6)
Ns_kept = [1000,2000,4000]
noises = [[9600,4800],[9600,6400],[12800,6400]]
seeds = [14142,41024,5096,1014,124095,12495,8952,50912,4095,8252]
n = 5
nInteractive = 5
studentConfThr = 0.5
threshold_ensemble = 0.4

#Data Labels
with open(os.path.join("data","all_preds_for_student.p"), "rb") as f:
    all_preds_for_student = pickle.load(f)

results = {} 
for j in range(5):
    #Get Data
    with open(os.path.join("data","data_student",str(N)+"_"+str(j)), "rb") as f:
        dataset = pickle.load(f)
    X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]
    preds = all_preds_for_student[str(N)+","+str(j)]
    #Eval
    n_classes = y_test.shape[1]
    res_dataset = {}
    for noise_c, noise in noises:
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
                #get data subset and partition initial student training data and data for interactive aggregation
                indices_subset = np.array(random.sample(list(range(len(y_train))), N_kept), dtype=int)
                preds_init = preds[:,indices_subset[:int(N_kept/2)],:]
                preds_rest = preds[:,indices_subset[int(N_kept/2):],:]
                X_init = X_train[indices_subset[:int(N_kept/2)],:]
                X_rest = X_train[indices_subset[int(N_kept/2):],:]
                y_rest = y_train[indices_subset[int(N_kept/2):],:]
                #Train students on init subset
                aggregated_preds_adapt, rdps_adapt, _ = aggregate_gaussian(preds_init, "adaptive", noise, orders, T, seed, ignore_privacy=False)
                aggregated_preds_fixed, rdps_fixed, _ = aggregate_gaussian(preds_init, fixed_threshold*N, noise, orders, T, seed, ignore_privacy=False)
                
                rdps_adapt = np.sum(rdps_adapt, axis=0)
                rdps_fixed = np.sum(rdps_fixed, axis=0)
                knn_init_adapt = MLkNN(k=10, s=1.0)
                knn_init_adapt.fit(X_init, aggregated_preds_adapt)
                knn_init_fixed = MLkNN(k=10, s=1.0)
                knn_init_fixed.fit(X_init, aggregated_preds_fixed)
                #adaptive threshold: Get aggregated preds for rest subset
                predictions_student = knn_init_adapt.predict_proba(X_rest).toarray()
                indices_interactive = []
                labels_interactive = np.zeros((1,predictions_student.shape[1]))
                for l in range(preds_rest.shape[1]):
                    pred_student = predictions_student[l,:]
                    aggregated_pred, rdp_interactive, rdp, reinforced = aggregate_gaussian_interactive(preds_rest[:,l,:], pred_student,"adaptive", threshold_ensemble*N, nInteractive, studentConfThr, noise_c,noise,T=T, n=n, orders=orders, seed=seed, ignore_privacy=False)
                    if not aggregated_pred is None:
                        labels_interactive = np.vstack((labels_interactive, np.expand_dims(aggregated_pred, axis=0)))
                        indices_interactive.append(l)
                    rdps_adapt += rdp_interactive + rdp
                labels_interactive = labels_interactive[1:,:]
                indices_interactive = np.array(indices_interactive)
                dps_adapt[k] = get_dp(rdps_adapt, orders, delta)
                #adaptive threshold: get shuffled concatenated data
                all_labels = np.vstack((aggregated_preds_adapt, labels_interactive))
                X_used = np.vstack((X_init, X_rest[indices_interactive]))
                X_used, all_labels= shuffle(X_used, all_labels,  random_state=seed)
                #adaptive threshold: train student model on full subset and get performances
                knn_full_adapt = MLkNN(k=10,s=1.0)
                knn_full_adapt.fit(X_used, all_labels)
                y_pred_adapt = knn_full_adapt.predict(X_test)
                f1s_adapt[k] = f1_score(y_test, y_pred_adapt, average="macro", zero_division=1.0)
                pres_adapt[k] = precision_score(y_test, y_pred_adapt, average="macro", zero_division=0.0)
                recs_adapt[k] = recall_score(y_test, y_pred_adapt, average="macro", zero_division=1.0)

                #fixed threshold: Get aggregated preds for rest subset
                predictions_student = knn_init_fixed.predict_proba(X_rest).toarray()
                indices_interactive = []
                labels_interactive = np.zeros((1,predictions_student.shape[1]))
                for l in range(preds_rest.shape[1]):
                    pred_student = predictions_student[l,:]
                    aggregated_pred, rdp_interactive, rdp, reinforced = aggregate_gaussian_interactive(preds_rest[:,l,:], pred_student,fixed_threshold*N, threshold_ensemble*N, nInteractive, studentConfThr, noise_c,noise,T=T, n=n, orders=orders, seed=seed, ignore_privacy=False)
                    if not aggregated_pred is None:
                        labels_interactive = np.vstack((labels_interactive, np.expand_dims(aggregated_pred, axis=0)))
                        indices_interactive.append(l)
                    rdps_fixed += rdp_interactive + rdp
                labels_interactive = labels_interactive[1:,:]
                indices_interactive = np.array(indices_interactive)
                dps_fixed[k] = get_dp(rdps_fixed, orders, delta)
                #fixed threshold: get shuffled concatenated data
                all_labels = np.vstack((aggregated_preds_fixed, labels_interactive))
                X_used = np.vstack((X_init, X_rest[indices_interactive]))
                X_used, all_labels= shuffle(X_used, all_labels,  random_state=seed)
                #fixed threshold: train student model on full subset and get performances
                knn_full_fixed = MLkNN(k=10,s=1.0)
                knn_full_fixed.fit(X_used, all_labels)
                y_pred_fixed = knn_full_fixed.predict(X_test)
                f1s_fixed[k] = f1_score(y_test, y_pred_fixed, average="macro", zero_division=1.0)
                pres_fixed[k] = precision_score(y_test, y_pred_fixed, average="macro", zero_division=0.0)
                recs_fixed[k] = recall_score(y_test, y_pred_fixed, average="macro", zero_division=1.0)



            #Get median dataset index according to f1*dp
            f1dp_adapt = np.multiply(f1s_adapt,dps_adapt)
            f1dp_fixed = np.multiply(f1s_fixed,dps_fixed)
            idx_adapt = np.argsort(f1dp_adapt)[len(f1dp_adapt)//2]
            idx_fixed = np.argsort(f1dp_fixed)[len(f1dp_fixed)//2]

            res_noise["N_kept="+str(N_kept)] = {
                "results_adapt": [pres_adapt[idx_adapt],recs_adapt[idx_adapt],f1s_adapt[idx_adapt], dps_adapt[idx_adapt]],
                "results_fixed": [pres_fixed[idx_fixed],recs_fixed[idx_fixed],f1s_fixed[idx_fixed], dps_fixed[idx_fixed]]
            }
        res_dataset["noise_c="+str(noise_c)+", noise="+str(noise)] = res_noise
    results[str(N)+"_"+str(j)] = res_dataset
pickle.dump(results,open(os.path.join("results/","paper_get_student_performance_interactive_all_noises_updatedPrivacyBound"),"wb"))
        
            
            
        