from generate_data import generate_data
import numpy as np

from sklearn.model_selection import train_test_split
import pickle
import os

#Parameters
N_per_teacher=1000#Remark: Different from before because we consider more teachers!
q = 30
M_rel = 14
M_irr = 7
M_red = 7
r_max = 0.8
r_min = ((float(q)/10)+1) / float(q)
noise = 0.05

seeds = [1234,1495,75329,5982,5149]
Ns = [50,100,300]
for N in Ns:
    N_data = (1.0/0.7) * N * N_per_teacher
    for i in range(len(seeds)):
        seed = seeds[i]
        #generate data
        points, labels = generate_data(r_min=r_min,r_max=r_max, q=q, M_rel=M_rel, M_irr=M_irr, M_red=M_red, N=N_data, noise=noise, seed=seed)
        #Split into train test data for teachers
        X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=0.3, random_state=seed)
        data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        pickle.dump(data,open(os.path.join('data_teachers',str(N)+"_"+str(i)),"wb"))

