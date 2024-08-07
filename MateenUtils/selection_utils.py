import numpy as np
import torch 
import torch.backends.cudnn as cudnn
from scipy.spatial.distance import pdist, squareform
import random 
import utils
from tqdm import tqdm

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"





def get_unique(reps, data_idx, min_distance):
    distances = squareform(pdist(reps))
    num_samples = distances.shape[0]
    valid = np.ones(num_samples, dtype=bool)
    counts = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        if valid[i]:
            neighbors = distances[i, :] < min_distance
            counts[i] = np.sum(neighbors) - 1  
            valid[neighbors] = False  
            valid[i] = True  
    filtered_reps = reps[valid]
    filtered_data_idx = np.array(data_idx)[valid]
    counts = counts[valid].tolist()
    return filtered_reps, filtered_data_idx, counts

def get_informative(model, data, data_idx, budget, retention_rate, initial_min_distance=0.1):
    data_idx = np.array(data_idx)
    data = torch.from_numpy(data).float().to(device)
    errs_vector = utils.getMSEvec(model(data), data).cpu().data.numpy()
    reps = model(data).cpu().data.numpy()
    reps = np.hstack((reps, errs_vector))
    target_count = int(retention_rate * len(data))
    lower_bound = 0
    upper_bound = 1
    tolerance = 0.0000001  
    max_iterations = 50 
    iteration = 0

    while (upper_bound - lower_bound) > tolerance and iteration < max_iterations:
        iteration += 1
        mid_point = (upper_bound + lower_bound) / 2
        filtered_reps, filtered_data_idx, similar_samples = get_unique(reps, data_idx, mid_point)
        if len(filtered_data_idx) < target_count:
            upper_bound = mid_point  
        else:
            lower_bound = mid_point 
    similar_samples = min_max_scaling(similar_samples)
    return filtered_data_idx, similar_samples

def min_max_scaling(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_rep(model, data, idx, budget, similar_rates, lambda_0, lambda_1=1.0):
    data = torch.from_numpy(data).float().to(device)
    reps = utils.getMSEvec(model(data), data).cpu().data.numpy()
    distances = squareform(pdist(reps))
    np.fill_diagonal(distances, 0)  
    distance_sums = distances.sum(axis=1)
    distance_sums = min_max_scaling(distance_sums)
    final_score = (lambda_0 * distance_sums) + (lambda_1 * similar_rates)
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


def mateen_selector(model, data, labels, args):    
    temp_idx = []
    batch_size = args.mini_batch_size
    if len(labels) > batch_size:
        batches_indices = data_to_bins(model, data, batch_size=batch_size)
    else:
        batches_indices = [np.arange(len(data))] 
    
    for batch in batches_indices:
        label_budget = int(args.selection_budget * len(batch))
        informative_idx, similar_rates = get_informative(model, data[batch], batch, args.selection_budget, args.retention_rate) 
        informative_idx = np.array(informative_idx)
        if len(informative_idx) > label_budget:
            selected_idx = get_rep(model, data[informative_idx], informative_idx, label_budget, similar_rates, args.lambda_0)
            temp_idx.extend(selected_idx)
        else:
            temp_idx.extend(informative_idx)
    temp_idx = np.array(temp_idx)    
    selected_idx = [idx for idx in temp_idx if labels[idx] == 0]
    
    if len(selected_idx) == 0:
        return None, temp_idx, labels[temp_idx]
    return data[selected_idx], temp_idx, labels[temp_idx]
