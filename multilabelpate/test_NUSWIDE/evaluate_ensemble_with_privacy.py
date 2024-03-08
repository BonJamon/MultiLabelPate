import numpy as np
import random
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import time, os, random
from Helper import NusDataset, Helperclass
from CNN_Networks import Dummy
from torch.utils.data import Subset
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelRecall, MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy
import pickle
from CNN_Networks import Resnext50
import io
from collections import OrderedDict
from multilabelpate.Aggregation import aggregate_gaussian

# Fix all seeds to make experiments reproducible
seed=2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

#threshold_ensemble = "adaptive"
noise_scales = np.array([2.0,4.0])
orders = np.array([1.000001,2,4,8,16,32])
np.set_printoptions(threshold=np.inf)

batch_size = 32
result_path = 'results/'
INDIVIDUAL_TEACHER_THRESHOLD = 0.5
t = 4

    
def get_dp(rdps, orders, delta):
    rdps_sum = np.sum(rdps, axis=0)
    delta_term = np.divide(np.log(1.0 / delta), orders - 1)
    return np.min(rdps_sum + delta_term)


def get_aggregated_preds(num_teachers, threshold_ensemble_fixed):
    preds = Helperclass.get_predictions(num_teachers)
    preds_binary = np.where(preds.numpy() > INDIVIDUAL_TEACHER_THRESHOLD, 1.0, 0.0)
    
    aggregated_preds = []
    pers = []
    dps = []
    for i in range(noise_scales.shape[0]):
        noise_scale_ensemble = noise_scales[i]
        preds_aggr, rdps, per_dep = aggregate_gaussian(preds_binary, threshold_ensemble_fixed, noise_scale_ensemble, orders, T=t,seed=seed,ignore_privacy=False)
        best_dp = get_dp(rdps, orders, delta)
        best_dp = best_dp / rdps.shape[0] #get dp per datapoint
        aggregated_preds.append(preds_aggr)
        pers.append(per_dep)#%rdp_dependend used
        dps.append(best_dp)
    return aggregated_preds, dps, pers

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

if __name__ == '__main__':
    #make checkpoint folder if not existing
    #Get Data
    test_data, train_data = NusDataset.preprocessing()
    data_length = len(test_data)
    delta = 1.0 / len(train_data)
    no_classes = len(test_data.classes)
    #Get Labels and Metrics
    labels = Helperclass.get_labels()
    #get metrics
    metric_map = MultilabelAveragePrecision(num_labels=no_classes, average='macro', threshold=0.5)
    metric_pre_ma = MultilabelPrecision(num_labels=no_classes, average='macro', threshold=0.5)
    metric_re_ma = MultilabelRecall(num_labels=no_classes, average='macro', threshold=0.5)
    metric_f1_ma = MultilabelF1Score(num_labels=no_classes, average='macro', threshold=0.5)
    metric_acc_ma = MultilabelAccuracy(num_labels=no_classes, average='macro', threshold=0.5)
    metric_pre_mi = MultilabelPrecision(num_labels=no_classes, average='micro', threshold=0.5)
    metric_re_mi = MultilabelRecall(num_labels=no_classes, average='micro', threshold=0.5)
    metric_f1_mi = MultilabelF1Score(num_labels=no_classes, average='micro', threshold=0.5)
    metric_acc_mi = MultilabelAccuracy(num_labels=no_classes, average='micro', threshold=0.5)
    metrics = [metric_map, metric_pre_ma, metric_re_ma, metric_f1_ma, metric_acc_ma, metric_pre_mi, metric_re_mi, metric_f1_mi, metric_acc_mi]

    results = [{"order of performance metrics": "metric_map, metric_pre_ma, metric_re_ma, metric_f1_ma, metric_acc_ma, metric_pre_mi, metric_re_mi, metric_f1_mi, metric_acc_mi"}]
    results.append({"parameters": "threshold_ensemble_fixed: n_teacher / 3"})

    #For each teacher-combination compute performances
    teacher_to_test = [2,4,8,16]
    for i in range(len(teacher_to_test)):
        #get aggregated performance: fixed threshold
        threshold_ensemble_fixed = teacher_to_test[i] / 3  #fixed threshold
        aggregated_preds, dps, pers = get_aggregated_preds(teacher_to_test[i], threshold_ensemble_fixed)
        
        for j in range(noise_scales.shape[0]):
            preds = torch.from_numpy(aggregated_preds[j])
            performances = np.zeros(len(metrics))
            for k in range(len(metrics)):
                metric = metrics[k]
                performances[k] = metric(preds=preds, target=labels.int())

            res_dict = {"num_teacher": teacher_to_test[i], "noise scale":noise_scales[j], "performances": performances, "privacy costs": dps[j],"per_dep": pers[j]}
            results.append(res_dict)
    pickle.dump(results,open(os.path.join("results/","privacy_performance_costs.p"),"wb"))
    