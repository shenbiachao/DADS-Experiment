# -*- coding: utf-8 -*-

"""
Apr 30: 对Semi-supervised AD实验重新设计的Loader


"""

#------------------------------------------------------------------------------#
#                                 MODULE                                   #
#------------------------------------------------------------------------------#

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import argparse

from __init__ import logger
from torch.utils.data import Dataset as PytorchDataset


#------------------------------------------------------------------------------#
#                                 PARAMETERS                                   #
#------------------------------------------------------------------------------#

## Hyperparameter
seed = 42

## 10% of TOTAL black train data are labeled
## If number of black samples in training dataset is 50, 0.1 means 5 labeled black dataset is available in training set
## the rest of the black dataset(45 out of 50) will be discarded（unless COMTAINATION_RATIO>0）.
ANOMALIES_FRACTION =  0.1 

## Labeled normalies to labeled anomalies ratio: 
## Say 5 out of 50 labeled black samples in the training set, NORMALIES_RATIO=5 means 25 labeled normal samples will be in the dataset
## NORMALIES_RATIO=0 means no labeled normal samples in the training dataset.
NORMALIES_RATIO= 5

## Proportion of unlabeled black sampleds in the unlabeled training dataset
## If unlabeled training size = 100, COMTAINATION_RATIO=0.01 means 1 out of 100 is black sample.
COMTAINATION_RATIO = 0.01


#------------------------------------------------------------------------------#
#                                 MAIN                                   #
#------------------------------------------------------------------------------#


class TabularData(PytorchDataset):

    """

    The class represents Tabular Anomlay Detection Dataset Module.
    The Dataset is inherited from PytorchDataset and CSVDataSets

    Arguments:
    ---------
        dataset_name(str):
            Dataset name
        _dataset(pd.DataFrame):
            pd.Dataframe includes raw image path, processed image path, ETL results.
            For training, ground truth is saved in "MUBADA" column as well.

    Methods:
    --------

        from_path: 
            Load image data from path
        sanity_check: 
            input data sanity check

    Note: Split/Sanity Check/Consistency Check/Feature Check functions can be user-defined

    """
    def __init__(self, 
        dataset=None, 
        dataset_name='', 
        training=False, 
        target_col_list = None,
        numerical_feat_idx='',
        categorical_feat_idx=''
        ):

        
        self._dataset = dataset
        self.dataset_name = dataset_name
        self.numerical_feat_idx = numerical_feat_idx
        self.categorical_feat_idx = categorical_feat_idx
        self.target_col_list = target_col_list


    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        pass
        
    def set_label(self, label = 0):
        if 'label' in self._dataset.columns:
            logger.info("Label will be overwritten to {}".format(label))

        self._dataset['label'] = label

    @classmethod
    def load(cls, dataset_name='arrhythmia',training=True):
        """
        Load Anomaly Detection Benchmark datasets from ODPS label pool

        """

        if dataset_name in ['arrhythmia','annthyroid','cardio', 'shuttle','satimage2','satellite']:
            df = pd.read_csv("./data/{}.csv".format(dataset_name))

            ## unify label class name
            if 'label' not in df.columns and 'Class' in df.columns:
                df.rename({"Class":"label"}, axis=1, inplace=True)


        return cls(dataset=df,
            dataset_name=dataset_name, numerical_feat_idx='', 
            categorical_feat_idx='',training=training)


    
    def binary_anomaly_detection_label(self):
        """
        Make multi-class anomaly detection dataset as binary label dataset
        0 represents normal class and 1 represents 1 class
        """

        pass

    @staticmethod
    def semi_supervised_ad_sampling(
        df,seed = 42, anomalies_fraction = 0.1, comtaination_ratio = 0.01):

        """
        PU Learning Setting for semi-supervised anomaly detection experiment
        
        对数据集按80/20 分train/test
        对分好的train dataset里的黑标， 按anomalies_fraction*black_label_size，当作labeled 数据(labeled数据只可能有黑标)
        对分好的train dataset里的白标， 添加一定比例的黑标做为unlabeled dataset
        最后的train_dataset是上两步的集合(labeled black + unlabeled white + unlabeled black(optional))
        test dataset 还是一开始分到的20%原始数据
        
        Arguments:
        ---------
            anomalies_fraction(float): 
                按anomalies_fraction*black_label_size，采样黑标当作labeled 数据
            comtaination_ratio(float):
                对unlabeled train dataset, 采样train dataset里白标数量*comtaination_ratio的黑标添加到unlabeled train dataset
                e.g: train dataset 有100个白标and comtaination_ratio=0.05, 最后的unlabeled train dataset会是100白标+5黑标=105
                Note: comtaination_ratio超过原始数据黑标浓度话，会替换成原始黑标浓度

        returns:
        --------
            train_df: 按semi supervised AD setting采样后的train df
            test_df: 原始数据的20% stratified split


        """

        from sklearn.model_selection import train_test_split

        ## check comtaination rate
        odds = df['label'].sum()/len(df)  ## 原始数据黑标浓度

        y = df[['label']]
        X = df.drop(['label'],axis=1)

        ## Step 1: Stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(
                X, y,stratify=y, test_size=0.2,random_state=seed)

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        ## Step 2: Set up labeled positive label size with anomalies_fraction hyperparameters
        black_train_df = train_df.loc[train_df['label']==1]
        white_train_df = train_df.loc[train_df['label']==0]

        black_train_df_shuffled = black_train_df.sample(frac=1, random_state=seed)
        black_train_size = len(black_train_df_shuffled)
        labeled_black_train_df = black_train_df_shuffled.iloc[:int(anomalies_fraction*black_train_size)]

        ## Step 3: For rest of the black data in the training set, 
        ## use comtaination_ratio hyperparameter to add into unlabeled training set
        ## Note comtaination_ratio cannot be larger than odds(如果原数据最多就10%黑的，那comtaination_ratio不可能大于10%)

        comtaination_ratio = min(comtaination_ratio, odds)
        unlabeled_black_train_df = black_train_df_shuffled.iloc[int(anomalies_fraction*black_train_size):]
        white_train_size = len(white_train_df)
        unlabeled_black_train_size = int(white_train_size * comtaination_ratio)

        ## Add those into ublabeled data
        unlabeled_black_train_df2 = unlabeled_black_train_df.iloc[:unlabeled_black_train_size]

        ## Add unlabled black adata into unlabeld white train data
        unlabeled_train_df = pd.concat([unlabeled_black_train_df2,white_train_df])
        unlabeled_train_df['label'] = np.nan ## unlabeled data shouldn't have label column

        ## Step 4: finally: concat labeled black training data and comtainnated unlabeled data as final train size
        train_df2 = pd.concat([labeled_black_train_df, unlabeled_train_df])
        return train_df2, test_df

    @classmethod
    def concat_dataset(cls, dataset1, dataset2):
        """Concat two pd datasets together"""


        result_df = pd.concat([dataset1._dataset,dataset2._dataset])


        return cls(dataset=result_df)

def test():
    """
    Unit test semi_supervised dataloader

    """

    ## Hyperparameter
    seed = 42
    ANOMALIES_FRACTION =  0.1 ## 10% black train data are labeled
    COMTAINATION_RATIO = 0.01

    DATASET = "shuttle"

    ad_ds = TabularData.load('shuttle')
    df = ad_ds._dataset

    ## Semi-supervised setting output
    train_df, test_df = TabularData.semi_supervised_ad_sampling(
        df, seed = seed, anomalies_fraction = ANOMALIES_FRACTION, comtaination_ratio = COMTAINATION_RATIO
        )

if __name__ == '__main__':
    test()





