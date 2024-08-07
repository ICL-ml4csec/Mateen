import AE as model_base
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import random
import utils
import copy

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"



def get_best_models(mode, models, thresholds, data, y_true):
    f1_list = []
    for i in range(len(models)):
        y_pred, probs = utils.preds_and_probs(models[i], thresholds[i], data)
        f1 = f1_score(y_true, y_pred, average='micro')
        f1_list.append(f1)
    index_of_max = f1_list.index(max(f1_list))
    if mode == "selection":
        return models[index_of_max], thresholds[index_of_max], index_of_max, max(f1_list), f1_list 
    elif mode == "merge":
        return f1_list

def get_values_within_margin(lst, margin=0.10):
    max_value = max(lst)
    threshold = max_value * (1 - margin)
    indices = [i for i, value in enumerate(lst) if value >= threshold]
    return indices

def compute_val_f1(model, data, y_true, previous_data):
    thres = utils.threshold_calulation(model, previous_data)
    predictions, _ = utils.preds_and_probs(model, thres, data)
    f1 = f1_score(y_true, predictions, average='micro')
    return f1

def merge_layer_weights(layer1, layer2, alpha):
    layer2.weight.data = alpha * layer1.weight.data + (1 - alpha) * layer2.weight.data
    layer2.bias.data = alpha * layer1.bias.data + (1 - alpha) * layer2.bias.data


def print_model_layer_names(model):
    for name, _ in model.named_parameters():
        print(name)

def merge_models(model1: nn.Module, model2: nn.Module, data, y_true, previous_data):
    input_shape = data.shape[1]
    empty_model = model_base.autoencoder(input_shape)
    empty_model.to(device)
    best_alpha = 0.5  
    best_f1 = -float('inf')
    original_state_dict = model2.state_dict()  
    for alpha_value in [i * 0.01 for i in range(101)]: 
        alpha = torch.tensor(alpha_value).to(device)
        temp_model = copy.deepcopy(empty_model)
        temp_model.load_state_dict(original_state_dict) 
        for (seq_name1, seq1), (seq_name2, seq2) in zip(model1._modules.items(), temp_model._modules.items()):
            for (layer_name1, layer1), (layer_name2, layer2) in zip(seq1._modules.items(), seq2._modules.items()):
                if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
                    merge_layer_weights(layer1, layer2, alpha)

        f1 = compute_val_f1(temp_model, data, y_true, previous_data)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha_value 
            print(f' Alpha {alpha_value} -- F1: {f1}')

    print(f'Best Alpha is {best_alpha}')
    final_model = copy.deepcopy(empty_model)
    final_model.load_state_dict(original_state_dict) 
    for (seq_name1, seq1), (seq_name2, seq2) in zip(model1._modules.items(), final_model._modules.items()):
        for (layer_name1, layer1), (layer_name2, layer2) in zip(seq1._modules.items(), seq2._modules.items()):
            if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
                merge_layer_weights(layer1, layer2, torch.tensor(best_alpha).to(device))

    return final_model  

def merge_tmp_models(models, thresholds, data, y_true, previous_data):
    f1_scores = get_best_models("merge", models, thresholds, data, y_true)
    candidates_idx = get_values_within_margin(f1_scores, margin=0.10)
    if len(candidates_idx) == 1:
        return models[candidates_idx[0]]
    merged_model = models[candidates_idx[0]]
    for idx in candidates_idx[1:]:
        merged_model = merge_models(merged_model, models[idx],data, y_true, previous_data)
    
    print(f'Models {candidates_idx} Have Been Merged')
    return merged_model
