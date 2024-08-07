import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append('MateenUtils/')

import data_processing as dp
import utils
import main as Mateen_main

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default="IDS2017", choices=["IDS2017", "IDS2018", "Kitsune", "mKitsune", "rKitsune"])

parser.add_argument('--window_size', type=int, default=50000, choices=[10000, 50000, 100000])


parser.add_argument('--performance_thres', type=int, default=0.99, choices=[0.99, 0.95, 0.90, 0.85, 0.8])


parser.add_argument('--max_ensemble_length', type=int, default=3, choices=[3, 5, 7])

parser.add_argument('--selection_budget', type=int, default=0.01, choices=[0.005, 0.01, 0.05, 0.1])

parser.add_argument('--mini_batch_size', type=int, default=1000, choices=[500, 1000, 1500])

parser.add_argument('--retention_rate', type=int, default=0.3, choices=[0.3, 0.5, 0.9])

parser.add_argument('--lambda_0', type=int, default=0.1, choices=[0.1, 0.5, 1.0])

parser.add_argument('--shift_threshold', type=int, default=0.05, choices=[0.05, 0.1, 0.2])


args = parser.parse_args()


def main(args):
    x_train, x_test, y_train, y_test = dp.prepare_data(scenario=args.dataset_name)
    x_slice, y_slice = dp.partition_array(x_data=x_test, y_data=y_test, slice_size=args.window_size)
    predicitons, probs_list = Mateen_main.adaptive_ensemble(x_train, y_train, x_slice, y_slice, args)
    _ = utils.getResult(y_test, predicitons)
    auc_rocs = utils.auc_roc_in_chunks(y_test, probs_list, chunk_size=args.window_size)
    print(f' Average AUC-ROC: {np.mean(auc_rocs)}, STD: {np.std(auc_rocs)}')
    df = pd.DataFrame({'Probabilities': probs_list, 'Predictions': predicitons})
    df.to_csv(f'Results/{args.dataset_name}-{args.selection_budget}.csv', index=False)
    return
    

main(args)