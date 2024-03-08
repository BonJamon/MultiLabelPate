import numpy as np
import random
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import time, os, random
from Aggregation import calc_adaptive_threshold

from Helper import NusDataset, Helperclass
from CNN_Network import Dummy
from torch.utils.data import Subset
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelRecall, MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy
import pickle
from CNN_Network import Resnext50
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
noise_scale_ensemble = 1500   
np.set_printoptions(threshold=np.inf)

batch_size = 32
result_path = 'results/'
checkpoint_path = 'checkpoints/evaluationTeacherPerformance_allmetrics/'
orders = np.array([1.000001,2,4,8,16,32])#irrelevant here



def get_aggregated_preds(testloader, num_teachers, no_classes, data_length, save_path, device, threshold_ensemble_fixed):
    preds = torch.torch.zeros((num_teachers, data_length, no_classes), dtype=torch.float32)
    for i in range(num_teachers):
        torch.cuda.empty_cache()
        model = Resnext50(no_classes)
        f = os.path.join(save_path,'checkpoint-{:06d}.pth'.format(i))
        model.load_state_dict(torch.load(f))
        model = nn.DataParallel(model)
        model.to(device)
        predictions = Helperclass.predict(model, testloader, device)
        preds[i] = predictions

    preds_binary = np.where(preds.numpy() > 0.5, 1.0, 0.0)
    aggregated_preds_adapt = torch.from_numpy(aggregate_gaussian(preds_binary, "adaptive", 0.0, orders, T=no_classes,seed=seed,ignore_privacy=True))
    aggregated_preds_adapt = torch.from_numpy(aggregate_gaussian(preds_binary, threshold_ensemble_fixed, 0.0, orders, T=no_classes,seed=seed,ignore_privacy=True))
    return preds, aggregated_preds_adapt, aggregated_preds_fixed

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Get Data
    test_data, _ = NusDataset.preprocessing()
    data_length = len(test_data)
    no_classes = len(test_data.classes)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    #Get Labels and Metrics
    labels = torch.Tensor([])
    for _,label in testloader:
        labels = torch.cat((labels,label))
    sum_labels = torch.sum(labels, dim=1)
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
    print("got data", flush=True)

    results = [{"order of metrics": "metric_map, metric_pre_ma, metric_re_ma, metric_f1_ma, metric_acc_ma, metric_pre_mi, metric_re_mi, metric_f1_mi, metric_acc_mi"}]
    #compute results for single teacher
    model = Resnext50(no_classes) #load adapted ResNext50
    Helperclass.checkpoint_load(model, 'checkpoints/singleTeacher2/', 3)
    model = nn.DataParallel(model)
    model.to(device)
    preds_single = Helperclass.predict(model, testloader, device)
    res_single = np.zeros(len(metrics))
    for k in range(len(metrics)):
        metric = metrics[k]
        res_single[k] = metric(preds=preds_single, target=labels.int())
    res_dict = {"num_teacher": 1, "res_avg": res_single}
    results.append(res_dict)
    print("computed single teacher", flush=True)
    #For each teacher-combination compute performances
    teacher_to_test = [2,4,8,16]
    for i in range(len(teacher_to_test)):
        #get aggregated performance: fixed threshold
        print("teacher config: "+str(teacher_to_test[i]), flush=True)
        save_path = 'checkpoints/multipleTeacher'+str(teacher_to_test[i])+'/'
        threshold_ensemble_fixed = (teacher_to_test[i] // 3) * np.ones(data_length)#fixed threshold
        preds, aggregated_preds_adapt, aggregated_preds_fixed = get_aggregated_preds(testloader, teacher_to_test[i], no_classes, data_length, save_path, device, threshold_ensemble_fixed)
        
        #Get avg results for single teacher
        running_res_avg = np.zeros(len(metrics))
        for j in range(teacher_to_test[i]):
            for k in range(len(metrics)):
                metric = metrics[k]
                running_res_avg[k] += metric(preds=preds[j], target=labels.int())
        res_avg = running_res_avg / teacher_to_test[i]
        
        res_adapt = np.zeros(len(metrics))
        for k in range(len(metrics)):
            metric = metrics[k]
            res_adapt[k] = metric(preds=aggregated_preds_adapt, target=labels.int())
        
        res_fixed = np.zeros(len(metrics))
        for k in range(len(metrics)):
            metric = metrics[k]
            res_fixed[k] = metric(preds=aggregated_preds_fixed, target=labels.int())

        res_dict = {"num_teacher": teacher_to_test[i], "res_avg": res_avg, "res_adapt": res_adapt, "res_fixed": res_fixed}
        pickle.dump(res_dict,open(os.path.join(checkpoint_path,'teacher-{:06d}.p'.format(teacher_to_test[i])),"wb"))
        results.append(res_dict)
        print("one config done", flush=True)
    pickle.dump(results,open(os.path.join("results/","evaluationTeacherPerformance_allmetrics.p"),"wb"))
    