import numpy as np
from sklearn.metrics import f1_score
from collections import Counter
from datetime import datetime
import copy
import torch
import torch.nn as nn
import os
import gc
import pandas as pd
import random
import torch.backends.cudnn as cudnn
import model_utils as model_base
import data_processing as dp
import utils
import merge_utils as merge
import selection_utils as selection
from scipy.stats import ks_2samp
import sys
sys.path.append('../OWAD/Utils')
import AE

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"



def load_model(load_mode, input_shape, scenario, train_loader, data, num_epochs):
    if load_mode == "new":
        model, _ = AE.train(data, input_shape, epoches=100)
    else:
        model = torch.load(f'../Models/{scenario}.pth').to(device)
    return model
          
    
def ensemble_training(x_train, y_train=None, num_epochs=10, mode=None, scenario=1, load_mode=None):
    input_shape = x_train.shape[1]
    if mode == "init":
        train_loader, benign_train = dp.prepare_datasets(x_train, y_train)
    elif mode == None: 
        train_loader, _ = dp.loading_datasets(x_train)
    model = load_model(load_mode, input_shape, scenario, train_loader, x_train, num_epochs)
    return model
    
def model_update(x_train, y_train=None, num_epochs=100, model=None):
    input_shape = x_train.shape[1]
    train_loader, _ = dp.loading_datasets(x_train)
    model = model_base.train_autoencoder(model, train_loader, num_epochs=num_epochs, learning_rate=0.0001)
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


def CADE_selection(x_train, y_train, x_test, y_test, cade_model, budget):
    sys.path.append('CADEutils')
    import CADE_Selector as CADE_selection
    import Train_CADE as CADE_trainer
    x_selected, selected_idx, selected_true = CADE_selection.CADE_Selection(x_train, y_train, x_test, y_test, cade_model, budget=budget)
    x_train = np.concatenate((x_train, x_selected), axis=0)
    y_train = np.concatenate((y_train, selected_true))
    cade_model = CADE_trainer.retrain_cade(x_train, y_train, cade_model)
    x_selected = x_selected[selected_true==0]
    if len(x_selected) == 0:
        return None, selected_idx, selected_true, x_train, y_train, cade_model
    return x_selected, selected_idx, selected_true, x_train, y_train, cade_model


def select_and_adapt(probs, probs_vector, data_slice, label_slice, budget, performance_thres, models_list, threshold_list, benign_train, selection_method, selected_model, y_pred, max_ensemble_length, selected_threshold, x_train, y_train, cade_model=None):
    print(datetime.now())
    if selection_method == "mateen":
        x_selected, selected_idx, selected_true = selection.mateen_selector(selected_model, data_slice, label_slice, budget)
    elif selection_method == "uncertainty":
        x_selected, selected_idx, selected_true = selection.uncertainty_selector(data_slice, label_slice, probs, budget)
    elif selection_method == "owad":
        x_selected, selected_idx, selected_true = selection.owad_selector(selected_model, data_slice, label_slice, benign_train[-len(data_slice):], budget)
    elif selection_method == "cade":
        x_selected, selected_idx, selected_true, x_train, y_train, cade_model = CADE_selection(x_train, y_train, data_slice, label_slice, cade_model, budget)
    
        
    else: 
        print('selection method is incorrect') 
    print(datetime.now())
    print(f'Selected Predictions {Counter(y_pred[selected_idx])}')
    print(f'Selected True labels {Counter(selected_true.flatten())}')
    print(f' Predictions {Counter(y_pred)}')
    print(f' True labels {Counter(label_slice.flatten())}')
    performance = f1_score(selected_true, y_pred[selected_idx], average='micro')
    if (performance < performance_thres):  
        big_model = copy.deepcopy(models_list[0])
        print(f' Bad Performance: {performance}')
        if x_selected is not None:
            print(x_selected.shape)
            print(' Train Temp Model')
            benign_train = np.concatenate((benign_train, x_selected))
            #new_model =  model_update(data_slice, num_epochs=5, model=big_model) 
            # without finetuning...
            new_model =  model_update(x_selected, num_epochs=100, model=big_model) 
            #new_model, _ = AE.train(x_selected, x_selected.shape[1], epoches=100)
            
            thres = utils.threshold_calulation(new_model, benign_train) 
            models_list.append(new_model)
            threshold_list.append(thres)
            
            y_pred, _ = utils.preds_and_probs(models_list[0], threshold_list[0], data_slice[selected_idx])
            big_model_performance = f1_score(selected_true, y_pred,average='micro')
            if (big_model_performance < performance_thres):  
                print(f'Update Large Model (Current Performance {big_model_performance})')
                updated_model =  model_update(benign_train, num_epochs=10, model=models_list[0]) 
                updated_model_thres = utils.threshold_calulation(updated_model, benign_train) 
                models_list[0] = updated_model
                threshold_list[0] = updated_model_thres
            if len(models_list) >= max_ensemble_length:
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
    return models_list, threshold_list, selected_model, selected_threshold, benign_train, x_train, y_train, cade_model

def adaptive_ensemble(x_train, y_train, x_slice, y_slice, performance_thres=0.99, max_ensemble_length=5, budget=0.5, selection_method="mateen", shift_threshold=0.01, scenario=1):
    cade_model = None
    model = ensemble_training(x_train, y_train=y_train, num_epochs=100, mode="init", scenario=scenario) 
    benign_train = x_train[y_train==0]
    selected_threshold = utils.threshold_calulation(model, benign_train)
    if selection_method == "cade":
        cade_model = model_base.build_autoencoder(x_train.shape[1])
        cade_model = cade_model.to(device)
        cade_model.load_state_dict(torch.load(f'../Models/CADE-IDS17/Best.pth'))
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
        if isit_shift(old_probs, probs, shift_threshold) == True:
            probs_vector = utils.get_features_error(selected_model, x_slice[i])
            models_list, threshold_list, selected_model, selected_threshold, benign_train, x_train, y_train, cade_model = select_and_adapt(probs, probs_vector, data_slice, label_slice, budget, performance_thres, models_list, threshold_list, benign_train, selection_method, selected_model, y_pred, max_ensemble_length, selected_threshold, x_train, y_train, cade_model=cade_model)
    return predicitons, probs_list

def mateen(x_train, y_train, x_slice, y_slice, performance_thres, max_ensemble_length, budget, selection_method, scenario):
    predicitons, probs_list = adaptive_ensemble(x_train, y_train, x_slice, y_slice, performance_thres=performance_thres, max_ensemble_length=max_ensemble_length, budget=budget, selection_method=selection_method, scenario=scenario)
    return predicitons, probs_list


def single_model(x_train, y_train, x_slice, y_slice, performance_thres=0.99, budget=0.5, shift_threshold=0.01, scenario=1):
    model = ensemble_training(x_train, y_train=y_train, num_epochs=100, mode="init", scenario=scenario) 
    benign_train = x_train[y_train==0]
    selected_threshold = utils.threshold_calulation(model, benign_train)
    predicitons = []
    probs_list = []
    print(f'Updating Models Process Started!')
    for i in range(len(x_slice)):
        print(f'Step {i+1}/{len(x_slice)}')
        y_pred, probs = utils.preds_and_probs(model, selected_threshold, x_slice[i])
        _, old_probs = utils.preds_and_probs(model, selected_threshold, benign_train[-len(x_slice[i]):])
        predicitons.extend(y_pred)
        probs_list.extend(probs)
        data_slice = x_slice[i]
        label_slice = y_slice[i]
        if i+1 == len(x_slice):
            return predicitons, probs_list
        if isit_shift(old_probs, probs, shift_threshold) == True:
            probs_vector = utils.get_features_error(model, x_slice[i])
            x_selected, selected_idx, selected_true = selection.mateen_selector(model, data_slice, label_slice, budget)
            print(datetime.now())
            print(f'Selected Predictions {Counter(y_pred[selected_idx])}')
            print(f'Selected True labels {Counter(selected_true.flatten())}')
            print(f' Predictions {Counter(y_pred)}')
            print(f' True labels {Counter(label_slice.flatten())}')
            performance = f1_score(selected_true, y_pred[selected_idx], average='micro')
            if (performance < performance_thres):  
                print(f' Bad Performance: {performance}')
                if x_selected is not None:
                    print(x_selected.shape)
                    model =  model_update(benign_train, num_epochs=10, model=model) 
                    selected_threshold = utils.threshold_calulation(model, benign_train) 
    return predicitons, probs_list



from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score

def generate_monthly_filenames_np(start_year=2013, start_month=1, end_year=2018, end_month=12):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)

    current_date = start_date
    filenames = []

    while current_date <= end_date:
        filenames.append(current_date.strftime("%Y-%m"))
        current_date += timedelta(days=32)  # Ensure the next month is reached
        current_date = current_date.replace(day=1)  # Reset to the first day of the next month

    return np.array(filenames)  


def adaptive_ensemble_malware(performance_thres=0.99, max_ensemble_length=5, budget=0.5, selection_method="mateen", shift_threshold=0.005, scenario=1):
    cade_model = None
    model = torch.load(f'../Models/apigraph.pth').to(device)
    file = "../Datasets/data/gen_apigraph_drebin/2012-01to2012-12_selected.npz"
    loaded = np.load(file)
    loaded = np.load(file)
    x_train, y_train = loaded['X_train'], loaded['y_train']
    y_train = np.array([1 if item != 0 else 0 for item in y_train])
    benign_train = x_train[y_train==0]
    selected_threshold = utils.threshold_calulation(model, benign_train)
    predicitons = []
    probs_list = []
    auc_rocs = []
    y_true = []
    print(f'Updating Models Process Started!')
    models_list = [model]
    threshold_list = [selected_threshold]
    selected_model = model
    filenames_np = generate_monthly_filenames_np()
    
    for i in range(len(filenames_np)):
        file = f"../Datasets/data/gen_apigraph_drebin/{filenames_np[i]}_selected.npz"
        loaded = np.load(file)
        x_slice, y_slice = loaded['X_train'], loaded['y_train']
        y_slice = np.array([1 if item != 0 else 0 for item in y_slice])
        
        print(f'Step {i+1}/{len(filenames_np)}')
        y_pred, probs = utils.preds_and_probs(selected_model, selected_threshold, x_slice)
        _, old_probs = utils.preds_and_probs(selected_model, selected_threshold, benign_train[-len(x_slice):])
        predicitons.extend(y_pred)
        auc_roc = roc_auc_score(y_slice, probs)
        auc_rocs.append(auc_roc)
        y_true.extend(y_slice)
        probs_list.extend(probs)
        data_slice = x_slice
        label_slice = y_slice
        if i+1 == len(x_slice):
            return predicitons, probs_list
        if isit_shift(old_probs, probs, shift_threshold) == True:
            probs_vector = utils.get_features_error(selected_model, x_slice)
            models_list, threshold_list, selected_model, selected_threshold, benign_train, x_train, y_train, cade_model = select_and_adapt(probs, probs_vector, data_slice, label_slice, budget, performance_thres, models_list, threshold_list, benign_train, selection_method, selected_model, y_pred, max_ensemble_length, selected_threshold, x_train, y_train, cade_model=cade_model)
        print(f'Step {i+1}/{len(filenames_np)} -- AUC-ROC {auc_roc}')
    return predicitons, probs_list, auc_rocs, y_true

    