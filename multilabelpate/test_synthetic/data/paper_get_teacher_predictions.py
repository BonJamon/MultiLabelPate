
import pickle
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from Aggregation import aggregate_gaussian
from skmultilearn.adapt import MLkNN


Ns = [50,100,300]

all_preds_synthetic = {}
for N in Ns:
    for j in range(5):
        #Get data
        with open(os.path.join("data_teachers",str(N)+"_"+str(j)), "rb") as f:
            dataset = pickle.load(f)
        X_train, X_test, y_train, y_test = dataset["X_train"], dataset["X_test"], dataset["y_train"], dataset["y_test"]

        #train all teacher and report avg teacher performance
        train_length = X_train.shape[0] // N
        n_classes = y_train.shape[1]
        preds = np.zeros((N,X_test.shape[0], n_classes), dtype=np.float32)
        for k in range(N):
            #Get data subset
            X_train_k = X_train[k*train_length:(k+1)*train_length,:]
            y_train_k = y_train[k*train_length:(k+1)*train_length,:]
            #train classifier on subset
            knn = MLkNN(k=10, s=1.0)
            knn.fit(X_train_k, y_train_k)
            y_pred_k = knn.predict_proba(X_test)
            preds[k,:,:] = y_pred_k.toarray()
        all_preds_synthetic[str(N)+"_"+str(j)] = preds
pickle.dump(all_preds_synthetic, open("all_predictions.p", "wb"))