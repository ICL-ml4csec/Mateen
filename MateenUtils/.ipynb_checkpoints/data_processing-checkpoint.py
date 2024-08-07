import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch 
import os


def load_dataset(file_path, file_type=None):
    if file_type == 'parquet':
        return pd.read_parquet(file_path)
    else:
        return pd.read_csv(file_path)

def process_data(data):
    data = pd.DataFrame(data)
    return np.nan_to_num(data.astype(float))

def prepare_data(scenario=1):
    if scenario == 1:
        data_2017_path = "../Datasets/CICIDS2017/clean_data.csv"
        data_2017 = load_dataset(data_2017_path)
        train = data_2017[:693702]
        test = data_2017[693702:]
    elif scenario == 2:
        train_path = "../Datasets/IDS2018/TrainData.csv"
        test_path = "../Datasets/IDS2018/TestData.csv"
        train = load_dataset(train_path)
        test = load_dataset(test_path)
    elif scenario == 3:
        train_path = "../Datasets/Kitsune/TrainData.csv"
        test_path = "../Datasets/Kitsune/TestData.csv"
        train = load_dataset(train_path)
        print(f'Train Loaded with {len(train)} Samples')
        test = load_dataset(test_path)
        print(f'Test Loaded with {len(test)} Samples')
    elif scenario == 4:
        train_path = "../Datasets/Kitsune/TrainData.csv"
        test_path = "../Datasets/Kitsune/NewTestData.csv"
        train = load_dataset(train_path)
        print(f'Train Loaded with {len(train)} Samples')
        test = load_dataset(test_path)
        print(f'Test Loaded with {len(test)} Samples')
    elif scenario == 5:
        train_path = "../Datasets/IDS2018/TrainData.csv"
        test_path = "../Datasets/IDS2018/NewTestData.csv"
        train = load_dataset(train_path)
        print(f'Train Loaded with {len(train)} Samples')
        test = load_dataset(test_path)
        print(f'Test Loaded with {len(test)} Samples')
    elif scenario == 6:
        train_path = "../Datasets/Kitsune/TrainData.csv"
        test_path = "../Datasets/Kitsune/Recurring.csv"
        train = load_dataset(train_path)
        print(f'Train Loaded with {len(train)} Samples')
        test = load_dataset(test_path)
        print(f'Test Loaded with {len(test)} Samples')

    elif scenario == 7:
        train_path = "../Datasets/Kitsune/ExTrain.csv"
        test_path = "../Datasets/Kitsune/ExTest.csv"
        train = load_dataset(train_path)
        test = load_dataset(test_path)
    else:
        raise ValueError("Invalid scenario number.")

    print(f'Scenario {scenario} with: {len(train)} training samples and {len(test)} testing samples')
    y_train, x_train = train["Label"], process_data(train.drop('Label', axis=1))
    y_test, x_test = test["Label"], process_data(test.drop('Label', axis=1))
    scaler = MinMaxScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def partition_array(x_data=None, y_data=None, slice_size=50000):
    num_samples = x_data.shape[0]
    num_slices = num_samples // slice_size + 1
    
    x_slices = []
    y_slices = []
    for i in range(num_slices):
        start = i * slice_size
        end = min((i + 1) * slice_size, num_samples)
        x_slices.append(x_data[start:end])
        y_slices.append(y_data[start:end])
    print(f' Test data has been divided into slices of size {slice_size} and length of {len(x_slices)}')
    return x_slices, y_slices


def loading_datasets(benign_train):
    train_dataset = TensorDataset(torch.tensor(benign_train))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    return train_loader, benign_train

def prepare_datasets(x_train, y_train):
    indexes_ben_train = np.where(y_train == 0)[0]
    benign_train = x_train[indexes_ben_train]
    train_loader, benign_train = loading_datasets(benign_train)
    return train_loader, benign_train

def prepare_new_train_valid_data(x_train, new_set):
    if new_set is None:
        return x_train
    else:
        x_new_train = np.concatenate((x_train, new_set), axis=0)
    return x_new_train