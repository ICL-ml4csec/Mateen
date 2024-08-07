import torch
import numpy as np
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from collections import Counter


seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
device


getMSEvec = nn.MSELoss(reduction='none')

def se2rmse(a):
    return torch.sqrt(sum(a.t())/a.shape[1])

def threshold_calulation(model, x_data):
    model.eval()
    output = model((torch.tensor(x_data).float()).to(device))
    mse_vec = getMSEvec(output, torch.tensor(x_data).to(device))
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    thres = max(rmse_vec)
    rmse_vec.sort()
    pctg = 0.95
    thres = rmse_vec[int(len(rmse_vec)*pctg)]
    return thres
    
def preds_and_probs(model, threshold, X_test):
    X_test_tensor = torch.from_numpy(X_test).type(torch.float).to(device)
    model.eval()
    output = model(X_test_tensor)
    mse_vec = getMSEvec(output, X_test_tensor)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    y_pred = np.asarray([0] * len(rmse_vec))
    idx_mal = np.where(rmse_vec > threshold)
    y_pred[idx_mal] = 1
    return y_pred, rmse_vec


def get_features_error(model, X_test):
    X_test_tensor = torch.from_numpy(X_test).type(torch.float).to(device)
    model.eval()
    output = model(X_test_tensor).cpu().data.numpy()
    errors = np.abs(X_test - output)
    return errors


def getResult(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    positive_label = 0
    print("Predicted Labels", Counter(y_pred))
    print("True Labels", Counter(y_true))
    if positive_label == 0:
        print('Positive label: 0')
        tp, fn, fp, tn = cm.ravel()
    else:
        print('Positive label: 1')
        tn, fp, fn, tp = cm.ravel()
    attacks = tp + fn
    normals = fp + tn
    accuracy = ((tp + tn) / (attacks + normals)) * 100
    precision = (tp / (tp + fp)) * 100
    recall = (tp / (tp + fn)) * 100
    f1 = (2 * (((precision / 100)* (recall / 100)) / ((precision / 100) + (recall / 100)))) * 100
    tnr = (tn / (tn + fp)) * 100
    macro_recall = recall_score(y_true, y_pred, average='macro') * 100
    macro_precision = precision_score(y_true, y_pred, average='macro') * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred) * 100
    tpr = (tp / (tp + fn)) * 100

    print("General Accuracy: {:.4f}".format(accuracy))
    print("Recall: {:.4f}".format(recall))
    print("Precision: {:.4f}".format(precision))
    print("F1 Score: {:.4f}".format(f1))
    print("True Negative Rate: {:.4f}".format(tnr))
    print(f"True Positive Rate: {tpr:.2f}%")
    print("Macro Recall: {:.4f}".format(macro_recall))
    print("Macro Precision: {:.4f}".format(macro_precision))
    print("Macro F1 Score: {:.4f}".format(macro_f1))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))
    return accuracy, recall, precision, f1, tnr, macro_recall, macro_precision, macro_f1, balanced_accuracy


def auc_roc_in_chunks(y_test, probs_list, chunk_size=50000):
    num_chunks = len(y_test) // chunk_size + (1 if len(y_test) % chunk_size != 0 else 0)
    auc_roc_scores = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        y_test_chunk = y_test[start:end]
        probs_list_chunk = probs_list[start:end]
        if set(y_test_chunk) == {0, 1}:
            auc_roc = roc_auc_score(y_test_chunk, probs_list_chunk)
            auc_roc_scores.append(auc_roc)    
    return auc_roc_scores