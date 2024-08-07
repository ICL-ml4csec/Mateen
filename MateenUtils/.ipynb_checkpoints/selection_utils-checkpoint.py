import os
import sys
from collections import Counter
sys.path.append('../Mateen/OWADutils/')

from owad_shifthunter import ShiftHunter
import owad_myutils
from owad_calibrator import Calibrator

import numpy as np
import torch 
import torch.backends.cudnn as cudnn
from scipy.spatial.distance import pdist, squareform
import random 
import utils
from tqdm import tqdm
from collections import Counter
seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"



def calibrate_detect(model, x_control, x_slice, y_slice, budget):
    cb = Calibrator(x_control, method='Isotonic')
    x_con = torch.from_numpy(x_control).float().to(device)
    x_slice_tensor = torch.from_numpy(x_slice).float().to(device)
    control_probs = utils.getMSEvec(model(x_con), x_con)
    test_probs = utils.getMSEvec(model(x_slice_tensor), x_slice_tensor)
    rmse_c = utils.se2rmse(control_probs).cpu().data.numpy()
    rmse_t = utils.se2rmse(test_probs).cpu().data.numpy()
    rmse_c[np.isnan(rmse_c)] = 0  
    rmse_t[np.isnan(rmse_t)] = 0
    cb.set_calibrator(rmse_c, is_P_mal=True)
    sh = ShiftHunter(rmse_c, rmse_t, cb, ad_type='Tab')
    t = owad_myutils.get_params('ShiftDetector')['test_thres']
    ohd_bgt = int(len(x_slice_tensor) * budget)
    exp_result = sh.explainer(x_slice_tensor.cpu().data.numpy(), y_slice, label_num = ohd_bgt) 
    return exp_result['remain_X_tre'], exp_result['rem_idx']


def owad_selector(model, data, labels, x_control, budget):
    remain_X_tre, rem_idx = calibrate_detect(model, x_control, data, labels, budget)
    if len(remain_X_tre) == 0:
        return None, rem_idx, labels[rem_idx]
    y_uncertain = labels[rem_idx]
    return remain_X_tre, rem_idx, y_uncertain

def get_unique(latent_reps, data_idx, min_distance):
    distances = squareform(pdist(latent_reps))
    num_samples = distances.shape[0]
    valid = np.ones(num_samples, dtype=bool)
    counts = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        if valid[i]:
            neighbors = distances[i, :] < min_distance
            counts[i] = np.sum(neighbors) - 1  
            valid[neighbors] = False  
            valid[i] = True  
    filtered_latent_reps = latent_reps[valid]
    filtered_data_idx = np.array(data_idx)[valid]
    counts = counts[valid].tolist()
    return filtered_latent_reps, filtered_data_idx, counts

def get_informative(model, data, data_idx, budget, initial_min_distance=0.1):
    data_idx = np.array(data_idx)
    data = torch.from_numpy(data).float().to(device)
    errs_vector = utils.getMSEvec(model(data), data).cpu().data.numpy()
    latent_reps = model(data).cpu().data.numpy()
    latent_reps = np.hstack((latent_reps, errs_vector))
    target_count = int(0.3 * len(data))
    lower_bound = 0
    upper_bound = 1
    tolerance = 0.0000001  
    max_iterations = 50 
    iteration = 0

    while (upper_bound - lower_bound) > tolerance and iteration < max_iterations:
        iteration += 1
        mid_point = (upper_bound + lower_bound) / 2
        filtered_latent_reps, filtered_data_idx, similar_samples = get_unique(latent_reps, data_idx, mid_point)
        if len(filtered_data_idx) < target_count:
            upper_bound = mid_point  
        else:
            lower_bound = mid_point 
    similar_samples = min_max_scaling(similar_samples)
    return filtered_data_idx, similar_samples

def min_max_scaling(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_rep(model, data, idx, budget, similar_rates, lambda_1=0.1, lambda_2=1.0):
    data = torch.from_numpy(data).float().to(device)
    latent_reps = utils.getMSEvec(model(data), data).cpu().data.numpy()
    distances = squareform(pdist(latent_reps))
    np.fill_diagonal(distances, 0)  
    distance_sums = distances.sum(axis=1)
    distance_sums = min_max_scaling(distance_sums)
    final_score = (lambda_1 * distance_sums) + (lambda_2 * similar_rates)
    sorted_indices = np.argsort(-final_score) 
    selected_data_idx_in_sorted = sorted_indices[:budget]
    selected_original_idx = idx[selected_data_idx_in_sorted]
    selected_data = data[selected_data_idx_in_sorted]
    return selected_original_idx

def data_to_bins(model, data, batch_size=1000):
    data = torch.from_numpy(data).float().to(device)
    recon_errs = utils.se2rmse(utils.getMSEvec(model(data), data)).cpu().data.numpy()
    indices = np.arange(len(recon_errs))
    sorted_indices = indices[np.argsort(recon_errs)[::-1]]  
    num_batches = len(sorted_indices) // batch_size
    batches_indices = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batches_indices.append(sorted_indices[start_idx:end_idx])
    return batches_indices


def noise_function(noise_rate, selected_idx, labels):
    
    # Ensure that the input is a numpy array for easier manipulation
    labels = np.array(labels)
    
    # Get indices of labels that are 1 and 0 within selected_idx
    ones_idx = [idx for idx in selected_idx if labels[idx] == 1]
    zeros_idx = [idx for idx in selected_idx if labels[idx] == 0]
    num_ones_to_flip = int(len(ones_idx) * noise_rate)
    num_zeros_to_flip = int(len(zeros_idx) * noise_rate)
    flip_ones_idx = np.random.choice(ones_idx, num_ones_to_flip, replace=False)
    flip_zeros_idx = np.random.choice(zeros_idx, num_zeros_to_flip, replace=False)
    flip_ones_idx = flip_ones_idx.astype(int)
    flip_zeros_idx = flip_zeros_idx.astype(int)
    labels[flip_ones_idx] = 0
    labels[flip_zeros_idx] = 1
    
    print(f'Noise Rate {noise_rate}; Benign Flipped {num_zeros_to_flip}; Malicious Flipped {num_ones_to_flip}')

    return labels

def mateen_selector(model, data, labels, budget, batch_size=1000):    
    temp_idx = []
    
    if len(labels) > batch_size:
        batches_indices = data_to_bins(model, data, batch_size=batch_size)
    else:
        batches_indices = [np.arange(len(data))] 
    
    for batch in batches_indices:
        label_budget = int(budget * len(batch))
        informative_idx, similar_rates = get_informative(model, data[batch], batch, budget)
        informative_idx = np.array(informative_idx)
        if len(informative_idx) > label_budget:
            selected_idx = get_rep(model, data[informative_idx], informative_idx, label_budget, similar_rates)
            temp_idx.extend(selected_idx)
        else:
            temp_idx.extend(informative_idx)
    temp_idx = np.array(temp_idx)    
    if len(temp_idx) == 0 and len(batch) > 0:  # Check to ensure batch is not empty
        temp_idx = np.random.choice(batch, size=int(budget * len(batch)), replace=False)
    
    print(f' Labels Before {Counter(labels)}')
    labels = noise_function(0.5, temp_idx, labels)
    print(f' Labels After {Counter(labels)}')
    selected_idx = [idx for idx in temp_idx if labels[idx] == 0]
    
    if len(selected_idx) == 0:
        return None, temp_idx, labels[temp_idx]
    
    return data[selected_idx], temp_idx, labels[temp_idx]


def uncertainty_selector(data, labels, recon_err, budget):
    budget = int(len(data) * budget)
    sorted_indices = np.argsort(recon_err)[::-1] 
    top_k_indices = sorted_indices[:budget]
    current_uncertain_indices = top_k_indices[labels[top_k_indices] == 0]
    if len(current_uncertain_indices) == 0:
        return None, top_k_indices, labels[top_k_indices]
    x_uncertain = data[current_uncertain_indices]
    y_uncertain = labels[top_k_indices]
    return x_uncertain, top_k_indices, y_uncertain
