# -*- coding: utf-8 -*-

"""
Apr 30: Semi-supervised anomaly detection experiment


"""

#------------------------------------------------------------------------------#
#                                 MODULE                                   #
#------------------------------------------------------------------------------#

import sys
import glob
import toml
from datetime import datetime

import pandas as pd
from tqdm import tqdm


from src.models.builder import BenchmarkBuilder
from src.models.ssad import SSAD
from src.datasets.semi_supervised_ad_loader import TabularData
# from semi_supervised_ad_loader import TabularData

#------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
#------------------------------------------------------------------------------#

MODEL_NAME = "devnet"

CONFIG_LIST = glob.glob("./config/{}/*.toml".format(MODEL_NAME))
CONFIG = toml.load(CONFIG_LIST)


## Hyperparameter

## 10% of TOTAL black train data are labeled
## If number of black samples in training dataset is 50, 0.1 means 5 labeled black dataset is available in training set
## the rest of the black dataset(45 out of 50) will be discarded（unless COMTAINATION_RATIO>0）.
ANOMALIES_FRACTION = CONFIG['SEMI_SUPERVISED_SETTING']['ANOMALIES_FRACTION']

## Labeled normalies to labeled anomalies ratio: 
## Say 5 out of 50 labeled black samples in the training set, NORMALIES_RATIO=5 means 25 labeled normal samples will be in the dataset
## NORMALIES_RATIO=0 means no labeled normal samples in the training dataset.
NORMALIES_RATIO = CONFIG['SEMI_SUPERVISED_SETTING']['NORMALIES_RATIO']

## Proportion of unlabeled black sampleds in the unlabeled training dataset
## If unlabeled training size = 100, COMTAINATION_RATIO=0.01 means 1 out of 100 is black sample.
COMTAINATION_RATIO = CONFIG['SEMI_SUPERVISED_SETTING']['COMTAINATION_RATIO']



#------------------------------------------------------------------------------#
#                                 MAIN                                   #
#------------------------------------------------------------------------------#


def baseline():

    """
    Evaluate semi-supervised model on each dataset
    Iterate over anomalies fraction, normalies_ratio, and comtaination_ratio

    """

    import itertools

    ## Create experiments setting
    benchmark_datasets = CONFIG['DATA']['DATASETS']
    seeds = CONFIG['DATA']['SEEDS']

    configs = list(itertools.product(
        benchmark_datasets,seeds, ANOMALIES_FRACTION,
        NORMALIES_RATIO, COMTAINATION_RATIO))

    
    results = []

    for config in tqdm(configs):

        ## Unpack hyperparameters
        dataset_name, seed, anomalies_fraction, normalies_ratio, comtaination_ratio = config

        ## Load data
        ad_ds = TabularData.load(dataset_name)
        df = ad_ds._dataset

        ## Semi-supervised setting
        if 'multi' in dataset_name:
            all_normal_classes = CONFIG['MULTI_CLASS_AD_SETTING']['NORMAL_CLASSES'][dataset_name]
            known_anomaly_class = CONFIG['MULTI_CLASS_AD_SETTING']['KNOWN_ANOMALY_CLASS'][dataset_name]

            train_df, val_df, test_df = TabularData.semi_supervised_multi_class_ad_sampling(
                df, seed = seed, anomalies_fraction = anomalies_fraction
                , normalies_ratio = normalies_ratio
                , comtaination_ratio = comtaination_ratio
                , all_normal_classes = all_normal_classes
                , known_anomaly_class = known_anomaly_class
                )

        else:
            train_df, val_df, test_df = TabularData.semi_supervised_ad_sampling(
                df, seed = seed, anomalies_fraction = anomalies_fraction
                , normalies_ratio = normalies_ratio
                , comtaination_ratio = comtaination_ratio
                )

        ## Build model
        model = BenchmarkBuilder.build(MODEL_NAME, CONFIG, seed=seed, dataset_name = dataset_name)
        # model = SSAD(CONFIG)

        ## Model training
        if MODEL_NAME == 'deepSAD':
            train_dataset = TabularData.load_from_dataframe(train_df,training=True)
            test_dataset = TabularData.load_from_dataframe(test_df,training=False)

            model.train(
                train_dataset = train_dataset, config=CONFIG
                )

            ## Model Evaluation
            roc_auc, roc_pr = model.evaluate(test_dataset)

            results.append([dataset_name,seed,
                anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])

        elif MODEL_NAME == "vime":

            roc_auc, roc_pr = model.train(train_df = train_df, val_df = test_df, config=CONFIG)
            results.append([dataset_name,seed,
                anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])

            del model

        else:
            model.train(
                train_df = train_df,
                val_df = val_df,
                config=CONFIG
                )

            ## Model Evaluation
            roc_auc, roc_pr = model.evaluate(test_df)

            results.append([dataset_name,seed,
                anomalies_fraction, normalies_ratio, comtaination_ratio, roc_auc, roc_pr])

    ## Save results
    results_df = pd.DataFrame(results)
    results_df.columns = ['dataset_name', 'seed', 'anomalies_fraction',
        'normalies_ratio', 'comtaination_ratio', 'roc_auc', 'roc_pr']

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    results_df.to_csv(
        # "./results/{}/{}_setting21_result.csv".format(MODEL_NAME,current_time), index=False)
        # "./results/{}/{}_setting12_rerun_result.csv".format(MODEL_NAME,current_time), index=False)
        "./results/{}/{}_setting22_multi_shuttle_rerun_result.csv".format(MODEL_NAME,current_time), index=False)

    pass


if __name__ == '__main__':
    baseline()


