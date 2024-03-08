import pickle
import os
import numpy as np
import random
from sklearn.metrics import f1_score, recall_score, precision_score
from multilabelpate.Aggregation import aggregate_gaussian, aggregate_gaussian_interactive
from multilabelpate.Privacy import compute_q_confidence_all, rdp_gaussian_all
from skmultilearn.adapt import MLkNN

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
Ns = [300]
orders = np.array([1.000001,2,4,8,16,32])
seeds = [1234,1495,75329,5982,5149]
delta  = 1.0 / (10**6)
noise_c = 9600
noise = 4800
n = 5

conf_breakpoints = [-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6]
nInteractives = [3,5,7]
studentConfThresholds = [0.5,0.6,0.7]

#get teacher predictions (=labels)
with open(os.path.join("data","all_predictions.p"), "rb") as f:
    all_preds_synthetic = pickle.load(f)


results = {} 
for i in range(len(Ns)):
    N = Ns[i]
    for j in range(5):
        seed = seeds[j]
        random.seed(seed)
        #Get data subsets
        N_kept = 2000
        with open(os.path.join("data","data_teachers",str(N)+"_"+str(j)), "rb") as f:
            dataset = pickle.load(f)
        X_test, y_test = dataset["X_test"],dataset["y_test"]
        preds = all_preds_synthetic[str(N)+"_"+str(j)]
        indices_subset = np.array(random.sample(list(range(len(y_test))), N_kept), dtype=int)
        preds_init = preds[:,indices_subset[:int(N_kept/2)],:]
        preds_rest = preds[:,indices_subset[int(N_kept/2):],:]
        X_init = X_test[indices_subset[:int(N_kept/2)],:]
        X_rest = X_test[indices_subset[int(N_kept/2):],:]
        y_rest = y_test[indices_subset[int(N_kept/2):],:]
        #Eval: For each combination nInteractive and studentConfThreshold
        n_classes = y_test.shape[1]
        res_dataset = {}
        for nInteractive in nInteractives:
            res_nInteractive = {}
            for studentConfThr in studentConfThresholds:
                res_studentConfThr = {}
                #Train student on init subset
                aggregated_preds_adapt, rdps_adapt, _ = aggregate_gaussian(preds_init, "adaptive", noise, orders, T, seed, ignore_privacy=False)
                aggregated_preds_fixed, rdps_fixed, _ = aggregate_gaussian(preds_init, fixed_threshold*N, noise, orders, T, seed, ignore_privacy=False)
                
                rdps_adapt = np.sum(rdps_adapt, axis=0)
                rdps_fixed = np.sum(rdps_fixed, axis=0)
                knn_init_adapt = MLkNN(k=10, s=1.0)
                knn_init_adapt.fit(X_init, aggregated_preds_adapt)
                knn_init_fixed = MLkNN(k=10, s=1.0)
                knn_init_fixed.fit(X_init, aggregated_preds_fixed)
                #Use interactive aggregator on rest of data and save performances
                for k in range(len(conf_breakpoints)-1):
                    #init performances
                    rdps_sum_adapt = np.copy(rdps_adapt)
                    rdps_sum_fixed = np.copy(rdps_fixed)
                    performances_aggr_adapt = np.zeros(3)
                    performances_aggr_fixed = np.zeros(3)
                    performances_student_adapt = np.zeros(3)
                    performances_student_fixed = np.zeros(3)
                    #adaptive threshold
                    #Need to iterate over all datapoints
                    predictions_student = knn_init_adapt.predict_proba(X_rest).toarray()
                    indices_teacher = []
                    indices_student = []
                    preds_student = np.zeros((1,predictions_student.shape[1]))
                    preds_teacher = np.zeros((1,predictions_student.shape[1]))
                    #get aggregated predictions
                    for l in range(preds_rest.shape[1]):
                        pred_student = predictions_student[l,:]
                        aggregated_pred, rdp_interactive, rdp, reinforced = aggregate_gaussian_interactive(preds_rest[:,l,:], pred_student,"adaptive", conf_breakpoints[k]*N, nInteractive, studentConfThr, noise_c,noise,T=T, n=n, orders=orders, seed=seed, ignore_privacy=False)
                        if not aggregated_pred is None:
                            if reinforced:
                                preds_student = np.vstack((preds_student, np.expand_dims(aggregated_pred, axis=0)))
                                indices_student.append(l)
                            else:
                                preds_teacher = np.vstack((preds_teacher, np.expand_dims(aggregated_pred, axis=0)))
                                indices_teacher.append(l)
                        rdps_sum_adapt += rdp_interactive + rdp
                    preds_student = preds_student[1:,:]
                    preds_teacher = preds_teacher[1:,:]
                    #report performances etc
                    indices_teacher = np.array(indices_teacher)
                    indices_student = np.array(indices_student)
                    dp_adapt = get_dp(rdps_sum_adapt, orders, delta)
                    pers_crossed_aggr_adapt = preds_teacher.shape[0]
                    pers_crossed_stud_adapt = preds_student.shape[0]
                    if pers_crossed_aggr_adapt>0:
                        performances_aggr_adapt[2] = f1_score(y_rest[indices_teacher], preds_teacher, average="macro", zero_division=1.0)
                        performances_aggr_adapt[0] = precision_score(y_rest[indices_teacher], preds_teacher, average="macro", zero_division=0.0)
                        performances_aggr_adapt[1] = recall_score(y_rest[indices_teacher], preds_teacher, average="macro", zero_division=1.0)
                    if pers_crossed_stud_adapt>0:
                        performances_student_adapt[2] = f1_score(y_rest[indices_student], preds_student, average="macro", zero_division=1.0)
                        performances_student_adapt[0] = precision_score(y_rest[indices_student], preds_student, average="macro", zero_division=0.0)
                        performances_student_adapt[1] = recall_score(y_rest[indices_student], preds_student, average="macro", zero_division=1.0)
                    #fixed threshold
                    indices_teacher = []
                    indices_student = []
                    preds_student = np.zeros((1,predictions_student.shape[1]))
                    preds_teacher = np.zeros((1,predictions_student.shape[1]))
                    predictions_student = knn_init_fixed.predict_proba(X_rest).toarray()
                    #get aggregated predictions
                    for l in range(preds_rest.shape[1]):
                        pred_student = predictions_student[l,:]
                        aggregated_pred, rdp_interactive, rdp, reinforced = aggregate_gaussian_interactive(preds_rest[:,l,:], pred_student,fixed_threshold*N, conf_breakpoints[k]*N, nInteractive, studentConfThr, noise_c,noise,T=T, n=n, orders=orders, seed=seed, ignore_privacy=False)
                        if not aggregated_pred is None:
                            if reinforced:
                                preds_student = np.vstack((preds_student, np.expand_dims(aggregated_pred, axis=0)))
                                indices_student.append(l)
                            else:
                                preds_teacher = np.vstack((preds_teacher, np.expand_dims(aggregated_pred, axis=0)))
                                indices_teacher.append(l)
                        rdps_sum_fixed += rdp_interactive + rdp
                    preds_student = preds_student[1:,:]
                    preds_teacher = preds_teacher[1:,:]
                    #report performances etc
                    indices_teacher = np.array(indices_teacher)
                    indices_student = np.array(indices_student)
                    dp_fixed = get_dp(rdps_sum_fixed, orders, delta)
                    pers_crossed_aggr_fixed = preds_teacher.shape[0]
                    pers_crossed_stud_fixed = preds_student.shape[0]
                    #Remark: pers_crossed for fixed threshold the same
                    if pers_crossed_aggr_fixed>0:
                        performances_aggr_fixed[2] = f1_score(y_rest[indices_teacher], preds_teacher, average="macro", zero_division=1.0)
                        performances_aggr_fixed[0] = precision_score(y_rest[indices_teacher], preds_teacher, average="macro", zero_division=0.0)
                        performances_aggr_fixed[1] = recall_score(y_rest[indices_teacher], preds_teacher, average="macro", zero_division=1.0)
                    if pers_crossed_stud_fixed>0:
                        performances_student_fixed[2] = f1_score(y_rest[indices_student], preds_student, average="macro", zero_division=1.0)
                        performances_student_fixed[0] = precision_score(y_rest[indices_student], preds_student, average="macro", zero_division=0.0)
                        performances_student_fixed[1] = recall_score(y_rest[indices_student], preds_student, average="macro", zero_division=1.0)


                    res_studentConfThr[str(conf_breakpoints[k])+"_"+str(conf_breakpoints[k+1])] = {"perPointsAggr adapt": pers_crossed_aggr_adapt,
                                                                                      "perPointsReinforced adapt": pers_crossed_stud_adapt,
                                                                                      "pre/rec/f1 aggr adapt": performances_aggr_adapt,
                                                                                      "pre/rec/f1 student adapt": performances_student_adapt,
                                                                                      "dp adapt": dp_adapt,
                                                                                      "perPointsAggr fixed": pers_crossed_aggr_fixed,
                                                                                      "perPointsReinforced fixed": pers_crossed_stud_fixed,
                                                                                      "pre/rec/f1 aggr fixed": performances_aggr_fixed,
                                                                                      "pre/rec/f1 student fixed": performances_student_fixed,
                                                                                      "dp fixed": dp_fixed
                                                                                      }
                res_nInteractive["confthr="+str(studentConfThr)] = res_studentConfThr
            res_dataset["nInt="+str(nInteractive)] = res_nInteractive
        results[str(N)+"_"+str(j)] = res_dataset
pickle.dump(results,open(os.path.join("results/","paper_analyseInteractiveCheck_updatedPrivacyBound"),"wb"))
        
            
            
        