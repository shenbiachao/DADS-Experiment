import numpy as np
import pandas as pd
import torch
import random
from sklearn.utils import shuffle
import umap
from config import Config

all_anomaly_classes = {'ann': [1, 2], 'cov': [4, 6], 'car': [2, 3], 'shu': [2, 3, 4, 5, 6, 7], 'har': [2, 3]}
all_normal_classes = {'ann': [3], 'cov': [2], 'car': [1], 'shu': [1], 'har': [1, 4, 5, 6]}
all_classes = {'ann': [3, 1, 2], 'cov': [2, 4, 6], 'car': [1, 2, 3], 'shu': [1, 2, 3, 4, 5, 6, 7], 'har': [1, 2, 3, 4, 5, 6]}

# load parameters in config.py
parameter = Config()
dataset_name = parameter.dataset_name
manual_dataset = parameter.manual_dataset
train_percentage = parameter.train_percentage
known_anomaly_num = parameter.known_anomaly_num
contamination_rate = parameter.contamination_rate
device = parameter.device
known_anomaly_classes = parameter.known_anomaly_classes
normalization = parameter.normalization


def load_original_data():
    """ Load original data
    supported dataset: annthyroid, covertype, cardio, shuttle
    @return dataset_a: initialization of anomaly dataset
    @return dataset_u: initialization of unlabeled dataset (temporary dataset is initialized as empty)
    @return dataset_test: dataset used for evaluation
    @return test_label: true label of dataset_test
    """
    if dataset_name == 'ann':
        source = pd.read_csv("data/annthyroid.csv")
        if normalization:
            source.iloc[:, :-1] = (source.iloc[:, :-1] - source.iloc[:, :-1].min()) / (
                    source.iloc[:, :-1].max() - source.iloc[:, :-1].min())
    elif dataset_name == 'cov':
        source = pd.read_csv("data/covertype.csv")
        if normalization:
            source.iloc[:, :10] = (source.iloc[:, :10] - source.iloc[:, :10].min()) / (
                    source.iloc[:, :10].max() - source.iloc[:, :10].min())
    elif dataset_name == 'car':
        source = pd.read_csv("data/cardio.csv")
        if normalization:
            source.iloc[:, :23] = (source.iloc[:, :23] - source.iloc[:, :23].min()) / (
                    source.iloc[:, :23].max() - source.iloc[:, :23].min())
    elif dataset_name == 'shu':
        source = pd.read_csv("data/shuttle.csv")
        if normalization:
            source.iloc[:, :9] = (source.iloc[:, :9] - source.iloc[:, :9].min()) / (
                    source.iloc[:, :9].max() - source.iloc[:, :9].min())
    else:
        assert 0, "Dataset not existed."
    known_anomaly_class = known_anomaly_classes[dataset_name]
    all_anomaly_class = all_anomaly_classes[dataset_name]
    source = shuffle(source)
    width = source.shape[1]
    length = len(source)
    dataset_train = source.iloc[:int(length*train_percentage), :]
    dataset_test = source.iloc[int(length*train_percentage):, :]

    dataset_a = pd.DataFrame(columns=source.columns)
    dataset_u = pd.DataFrame(columns=source.columns)
    for i in range(len(dataset_train)):
        label = dataset_train.iloc[i, width-1]
        if label == known_anomaly_class and len(dataset_a) < known_anomaly_num:
            dataset_a = dataset_a.append(dataset_train.iloc[i, :])
        else:
            if len(dataset_a) < known_anomaly_num:
                dataset_u = dataset_u.append(dataset_train.iloc[i, :])
            else:
                dataset_u = dataset_u.append(dataset_train.iloc[i:, :])
                break
    dataset_a = dataset_a.reset_index(drop=True)
    dataset_u = dataset_u.reset_index(drop=True)

    dataset_a = torch.tensor(dataset_a.values.astype(float))[:, :-1].float().to(device)
    dataset_u = torch.tensor(dataset_u.values.astype(float))[:, :-1].float().to(device)
    test_label = torch.tensor(dataset_test.values.astype(float))[:, -1].float().to(device)
    test_label = [1 if i in all_anomaly_class else 0 for i in test_label]
    dataset_test = torch.tensor(dataset_test.values.astype(float))[:, :-1].float().to(device)

    # source and index are used in function plot
    return dataset_a, dataset_u, dataset_test, test_label


def load_manual_data():
    """ Load data with anomaly percentage manually set
    supported dataset: har, covertype
    """
    if dataset_name == 'har':
        source = pd.read_csv("data/har.csv")
        if normalization:
            source.iloc[:, :-1] = (source.iloc[:, :-1] - source.iloc[:, :-1].min()) / (
                    source.iloc[:, :-1].max() - source.iloc[:, :-1].min())
    elif dataset_name == 'cov':
        source = pd.read_csv("data/covertype.csv")
        if normalization:
            source.iloc[:, :10] = (source.iloc[:, :10] - source.iloc[:, :10].min()) / (
                    source.iloc[:, :10].max() - source.iloc[:, :10].min())
    elif dataset_name == 'ann':
        source = pd.read_csv("data/annthyroid.csv")
        if normalization:
            source.iloc[:, :-1] = (source.iloc[:, :-1] - source.iloc[:, :-1].min()) / (
                    source.iloc[:, :-1].max() - source.iloc[:, :-1].min())
    else:
        assert 0, "Dataset not existed."

    source = shuffle(source)
    width = source.shape[1]

    dataset_a = pd.DataFrame(columns=source.columns)
    anomaly_temp = pd.DataFrame(columns=source.columns)
    normal_temp = pd.DataFrame(columns=source.columns)
    if dataset_name != 'cov':
        for i in range(len(source)):
            label = source.iloc[i, width - 1]
            if label == known_anomaly_classes[dataset_name] and len(dataset_a) < known_anomaly_num:
                dataset_a = dataset_a.append(source.iloc[i, :])
            elif label in all_anomaly_classes[dataset_name]:
                anomaly_temp = anomaly_temp.append(source.iloc[i, :])
            elif label in all_normal_classes[dataset_name]:
                normal_temp = normal_temp.append(source.iloc[i, :])
    else:
        normal_temp = source.iloc[:283302, :]
        anomaly_temp = source.iloc[283302:, :]
        dataset_a = anomaly_temp.sample(known_anomaly_num, replace=False)

    dataset_a = dataset_a.reset_index(drop=True)
    anomaly_temp= anomaly_temp.reset_index(drop=True)
    normal_temp = normal_temp.reset_index(drop=True)

    dataset_u = normal_temp.iloc[:int(len(normal_temp) * train_percentage), :]
    dataset_test = normal_temp.iloc[int(len(normal_temp) * train_percentage):, :]

    temp = 0
    for i in range(int(contamination_rate*len(dataset_u))+1):
        dataset_u = dataset_u.append(anomaly_temp.iloc[temp, :])
        temp = temp + 1
    for i in range(int(contamination_rate*len(dataset_test))+1):
        dataset_test = dataset_test.append(anomaly_temp.iloc[temp, :])
        temp = temp + 1
    dataset_u = dataset_u.reset_index(drop=True)
    dataset_test = dataset_test.reset_index(drop=True)

    dataset_a = torch.tensor(dataset_a.values.astype(float))[:, :-1].float().to(device)
    dataset_u = torch.tensor(dataset_u.values.astype(float))[:, :-1].float().to(device)
    test_label = torch.tensor(dataset_test.values.astype(float))[:, -1].float().to(device)
    test_label = [1 if i in all_anomaly_classes[dataset_name] else 0 for i in test_label]
    dataset_test = torch.tensor(dataset_test.values.astype(float))[:, :-1].float().to(device)

    return dataset_a, dataset_u, dataset_test, test_label


def load_data():
    if manual_dataset:
        dataset_a, dataset_u, dataset_test, test_label = load_manual_data()
    else:
        dataset_a, dataset_u, dataset_test, test_label = load_original_data()

    return dataset_a, dataset_u, dataset_test, test_label