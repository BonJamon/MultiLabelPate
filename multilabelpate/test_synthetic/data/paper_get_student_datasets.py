import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

Ns = [300]
random_state = 42

with open("all_predictions.p", "rb") as f:
    all_preds_synthetic = pickle.load(f)


all_preds_synthetic_for_student = {}
for N in Ns:
    for j in range(5):
        #Get data
        with open(os.path.join("data_teachers",str(N)+"_"+str(j)), "rb") as f:
            dataset = pickle.load(f)
        y_test = dataset["y_test"]
        X_test = dataset["X_test"]
        preds = all_preds_synthetic[str(N)+"_"+str(j)]
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_test, y_test, test_size=0.3, random_state=random_state)
        preds_train = np.zeros((N,y_train_new.shape[0], preds.shape[2]), dtype=np.float32)
        for n in range(N):
            preds_n = preds[n,:,:]
            preds_train_n, preds_test_n = train_test_split(preds_n, test_size=0.3, random_state=random_state)
            preds_train[n,:,:] = preds_train_n
        all_preds_synthetic_for_student[str(N)+","+str(j)] = preds_train
        data = {"X_train": X_train_new, "X_test": X_test_new, "y_train": y_train_new, "y_test": y_test_new}
        pickle.dump(data,open(os.path.join('data_student',str(N)+"_"+str(j)),"wb"))

pickle.dump(all_preds_synthetic_for_student, open('all_preds_for_student.p', 'wb'))
