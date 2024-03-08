import pickle
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from multilabelpate.Aggregation import aggregate_gaussian
from skmultilearn.adapt import MLkNN
#import t-approx


'''
For each dataset: 
1.) Report Base-Performance
2.) Avg Teacher Performance
3.) Aggregation Performance,
4.) Aggregation Performance with t-approximation

Aggregation Performances reported for adaptive and fixed threshold
'''
#Parameters
Ns = [50,100,300]
seeds = [1234,1495,75329,5982,5149]
T=15
fixed_threshold=0.3

#get teacher predictions (=labels)
with open(os.path.join("data","all_predictions.p"), "rb") as f:
    all_preds_synthetic = pickle.load(f)

results = {} 
for N in Ns:
    for j in range(5):
        #Get data
        with open(os.path.join("data","data_teachers",str(N)+"_"+str(j)), "rb") as f:
            dataset = pickle.load(f)
        X_train, X_test, y_train, y_test = dataset["X_train"], dataset["X_test"], dataset["y_train"], dataset["y_test"]

        #train single teacher
        knn = MLkNN(k=10, s=1.0)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=1.0)
        pre = precision_score(y_test, y_pred, average="macro", zero_division=0.0)#Which zerodivision? Just says it predicts no positive samples, but there could be some 
        rec = recall_score(y_test, y_pred, average="macro", zero_division=1.0)#which zerodivision? 1.0, cause there are no positive samples and it predicts None, so does everything right
       
        preds = all_preds_synthetic[str(N)+"_"+str(j)]
        preds_binary = preds >= 0.5
        #get avg teacher performance
        avg_f1 = 0.0
        avg_pre = 0.0
        avg_rec = 0.0
        for k in range(N):
            avg_f1 += f1_score(y_test, preds_binary[k,:,:], average="macro", zero_division=1.0)
            avg_pre += precision_score(y_test, preds_binary[k,:,:], average="macro", zero_division=0.0)
            avg_rec += recall_score(y_test, preds_binary[k,:,:], average="macro", zero_division=1.0)
        avg_f1 /= N
        avg_pre /= N
        avg_rec /= N

        orders = [1.0]#irrelevant
        seed = seeds[j]
        n_classes = y_train.shape[1]
        #Get aggregation Performances without t-approx
        #for adaptive
        aggregated_preds = aggregate_gaussian(preds, "adaptive", 0.0, orders, T=n_classes,seed=seed,ignore_privacy=True)#T=n_classes means no t-approx
        aggr_f1 = f1_score(y_test, aggregated_preds, average="macro", zero_division=1.0)
        aggr_pre = precision_score(y_test, aggregated_preds, average="macro", zero_division=0.0)
        aggr_rec = recall_score(y_test, aggregated_preds, average="macro", zero_division=1.0)
        
        #for fixed
        aggregated_preds_fixed = aggregate_gaussian(preds, fixed_threshold*N, 0.0, orders, T=n_classes,seed=seed,ignore_privacy=True)
        aggr_f1_fixed = f1_score(y_test, aggregated_preds, average="macro", zero_division=1.0)
        aggr_pre_fixed = precision_score(y_test, aggregated_preds, average="macro", zero_division=0.0)
        aggr_rec_fixed = recall_score(y_test, aggregated_preds, average="macro", zero_division=1.0)
        
        
        #Get aggregation Performances with t-approx
        #for adaptive
        aggregated_preds_T = aggregate_gaussian(preds, "adaptive", 0.0, orders, T=T,seed=seed,ignore_privacy=True)
        aggr_f1_T = f1_score(y_test, aggregated_preds_T, average="macro", zero_division=1.0)
        aggr_pre_T = precision_score(y_test, aggregated_preds_T, average="macro", zero_division=0.0)
        aggr_rec_T = recall_score(y_test, aggregated_preds_T, average="macro", zero_division=1.0)
        
        #for fixed
        aggregated_preds_fixed_T = aggregate_gaussian(preds, fixed_threshold*N, 0.0, orders, T=T,seed=seed,ignore_privacy=True)
        aggr_f1_fixed_T = f1_score(y_test, aggregated_preds_T, average="macro", zero_division=1.0)
        aggr_pre_fixed_T = precision_score(y_test, aggregated_preds_T, average="macro", zero_division=0.0)
        aggr_rec_fixed_T = recall_score(y_test, aggregated_preds_T, average="macro", zero_division=1.0)


        results[str(N)+"_"+str(j)] = {"single": [pre,rec,f1], 
                                      "avg": [avg_pre, avg_rec, avg_f1],
                                      "aggr adapt": [aggr_pre, aggr_rec, aggr_f1],
                                      "aggr fixed": [aggr_pre_fixed, aggr_rec_fixed, aggr_f1_fixed],
                                      "aggr adapt t-approx": [aggr_pre_T, aggr_rec_T, aggr_f1_T],
                                      "aggr fixed t-approx": [aggr_pre_fixed_T, aggr_rec_fixed_T, aggr_f1_fixed_T]}
        
pickle.dump(results,open(os.path.join("results/","paper_evaluateAggregation_final"),"wb"))
        



