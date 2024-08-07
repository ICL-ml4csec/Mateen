import numpy as np
from sklearn.metrics import f1_score
from collections import Counter
from datetime import datetime
import copy
import torch
import torch.nn as nn
import pandas as pd
import random
import torch.backends.cudnn as cudnn
import AE as model_base
import data_processing as dp
import utils
import merge_utils as merge
import selection_utils as selection
from scipy.stats import ks_2samp


seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_update(x_train, y_train=None, num_epochs=100, model=None):
    input_shape = x_train.shape[1]
    train_loader, _ = dp.loading_datasets(x_train)
    model = model_base.train_autoencoder(model, train_loader, num_epochs=num_epochs, learning_rate=0.0001)
    return model


def load_model(load_mode, input_shape, scenario, train_loader, data, num_epochs):
    if load_mode == "new":
        model = model_base.autoencoder(input_shape)
        model = model_update(data, num_epochs=num_epochs, model=model) 
    else:
        model = torch.load(f'Models/{scenario}.pth').to(device)
    return model
          
    
def ensemble_training(x_train, y_train=None, num_epochs=10, mode=None, scenario=1, load_mode=None):
    input_shape = x_train.shape[1]
    if mode == "init":
        train_loader, benign_train = dp.prepare_datasets(x_train, y_train)
    elif mode == None: 
        train_loader, _ = dp.loading_datasets(x_train)
    model = load_model(load_mode, input_shape, scenario, train_loader, x_train, num_epochs)
    return model
    

def isit_shift(recon_old, recon_new, threshold):
    recon_old_sorted = sorted(recon_old)
    recon_new_sorted = sorted(recon_new)
    ks_statistic, p_value = ks_2samp(recon_old_sorted, recon_new_sorted)
    if p_value < threshold:
        return True
    else:
        print(f' No Shift !')
        return False



def select_and_adapt(probs, probs_vector, data_slice, label_slice, models_list, threshold_list, benign_train, selected_model, y_pred, selected_threshold, x_train, y_train, args):
    print(datetime.now())
    x_selected, selected_idx, selected_true = selection.mateen_selector(selected_model, data_slice, label_slice, args)
    print(datetime.now())
    print(f'Selected Predictions {Counter(y_pred[selected_idx])}')
    print(f'Selected True labels {Counter(selected_true.flatten())}')
    print(f' Predictions {Counter(y_pred)}')
    print(f' True labels {Counter(label_slice.flatten())}')
    performance = f1_score(selected_true, y_pred[selected_idx], average='micro')
    if (performance < args.performance_thres):  
        big_model = copy.deepcopy(models_list[0])
        print(f' Bad Performance: {performance}')
        if x_selected is not None:
            print(x_selected.shape)
            print(' Train Temp Model')
            benign_train = np.concatenate((benign_train, x_selected))
            new_model =  model_update(x_selected, num_epochs=100, model=big_model) 
            thres = utils.threshold_calulation(new_model, benign_train) 
            models_list.append(new_model)
            threshold_list.append(thres)
            
            y_pred, _ = utils.preds_and_probs(models_list[0], threshold_list[0], data_slice[selected_idx])
            big_model_performance = f1_score(selected_true, y_pred,average='micro')
            if (big_model_performance < args.performance_thres):  
                print(f'Update Large Model (Current Performance {big_model_performance})')
                updated_model =  model_update(benign_train, num_epochs=10, model=models_list[0]) 
                updated_model_thres = utils.threshold_calulation(updated_model, benign_train) 
                models_list[0] = updated_model
                threshold_list[0] = updated_model_thres
            if len(models_list) >= args.max_ensemble_length:
                print('Cleaning Ensemble')
                print(f' Ensemble Length {len(models_list)}')
                temp_models = models_list[1:-1]
                temp_thresholds = threshold_list[1:-1]
                print(f' Merged Length {len(temp_models)}')
                temp_model = merge.merge_tmp_models(temp_models, temp_thresholds, data_slice[selected_idx], label_slice[selected_idx], benign_train)
                print('Fine Tune Merged Model')
                temp_model_thres = utils.threshold_calulation(temp_model, benign_train)
                models_list = [models_list[0], temp_model, models_list[-1]]
                threshold_list = [threshold_list[0], temp_model_thres, threshold_list[-1]]  
        selected_model, selected_threshold, model_idx, selected_f1, f1_list = merge.get_best_models("selection", models_list, threshold_list, data_slice[selected_idx], label_slice[selected_idx]) 
        print(f' Model {model_idx} Selected with F1 {selected_f1} ; other models F1s {f1_list}')
    return models_list, threshold_list, selected_model, selected_threshold, benign_train, x_train, y_train

def adaptive_ensemble(x_train, y_train, x_slice, y_slice, args):
    cade_model = None
    model = ensemble_training(x_train, y_train=y_train, num_epochs=100, mode="init", scenario=args.dataset_name) 
    benign_train = x_train[y_train==0]
    selected_threshold = utils.threshold_calulation(model, benign_train)
    predicitons = []
    probs_list = []
    print(f'Updating Models Process Started!')
    models_list = [model]
    threshold_list = [selected_threshold]
    selected_model = model
    for i in range(len(x_slice)):
        print(f'Step {i+1}/{len(x_slice)}')
        y_pred, probs = utils.preds_and_probs(selected_model, selected_threshold, x_slice[i])
        _, old_probs = utils.preds_and_probs(selected_model, selected_threshold, benign_train[-len(x_slice[i]):])
        predicitons.extend(y_pred)
        probs_list.extend(probs)
        data_slice = x_slice[i]
        label_slice = y_slice[i]
        if i+1 == len(x_slice):
            return predicitons, probs_list
        if isit_shift(old_probs, probs, args.shift_threshold) == True:
            probs_vector = utils.get_features_error(selected_model, x_slice[i])
            models_list, threshold_list, selected_model, selected_threshold, benign_train, x_train, y_train = select_and_adapt(probs, probs_vector, data_slice, label_slice, models_list, threshold_list, benign_train, selected_model, y_pred, selected_threshold, x_train, y_train, args)
    return predicitons, probs_list



