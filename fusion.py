#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:01:33 2024

@author: nestor
"""

import os
import numpy as numpy
import pickle
from constant import *
import pandas as pd
from utils import *
from models import *
from signal_analysis import SignalAnalysis

signal_type = 'M'
full_classes = True
LOSO = True

class_ammount = 'five' if full_classes else 'three'
cross_validation_type = 'LOSO' if LOSO else 'SGKFold'

INFORMATION_NAME = f'information_dataset_{signal_type}.pickle'
HANDCRAFTED_FEATURE_NAME = f'features_dataset_{signal_type}.pickle'
SPECTOGRAM_FEATURE_NAME = f'spectograms_dataset_{signal_type}.pickle'
CROSS_VALIDATION_NAME = f'folds_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FOLDS_NAME_SVM = f'best_results_folds_SVM_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_SVM = f'best_result_SVM_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FOLDS_NAME_DT = f'best_results_folds_DT_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_DT = f'best_result_DT_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FOLDS_NAME_RF = f'best_results_folds_RF_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_RF = f'best_result_RF_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'


PATH_HANDCRAFTED_FEATURE = os.path.join(RESULTS_PATH, HANDCRAFTED_FEATURE_NAME)
PATH_SPECTOGRAM_FEATURE = os.path.join(RESULTS_PATH, SPECTOGRAM_FEATURE_NAME)
PATH_CROSS_VALIDATION = os.path.join(RESULTS_PATH, CROSS_VALIDATION_NAME)
PATH_BEST_RESULTS_FOLDS_SVM = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_SVM)
PATH_BEST_RESULTS_FIXED_PARAMS_SVM = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_SVM)
PATH_BEST_RESULTS_FOLDS_DT = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_DT)
PATH_BEST_RESULTS_FIXED_PARAMS_DT = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_DT)
PATH_BEST_RESULTS_FOLDS_RF = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_RF)
PATH_BEST_RESULTS_FIXED_PARAMS_RF = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_RF)

info_dataframe = create_dataset(signal_type)

info_dataframe = info_dataframe[info_dataframe['label']!= '']
signal_analysis = SignalAnalysis(info_dataframe, signal_type)

with open(PATH_HANDCRAFTED_FEATURE, 'rb') as output:
    features_M = pickle.load(output)
with open(os.path.join(RESULTS_PATH, INFORMATION_NAME), 'rb') as output:
    info_dataframe_M = pickle.load(output)

#%% THROAT FEATURES FOR EARLY FUSSION

signal_type = 'T'
full_classes = True
LOSO = True

class_ammount = 'five' if full_classes else 'three'
cross_validation_type = 'LOSO' if LOSO else 'SGKFold'


INFORMATION_NAME = f'information_dataset_{signal_type}.pickle'
HANDCRAFTED_FEATURE_NAME = f'features_dataset_{signal_type}.pickle'
SPECTOGRAM_FEATURE_NAME = f'spectograms_dataset_{signal_type}.pickle'
CROSS_VALIDATION_NAME = f'folds_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FOLDS_NAME_SVM = f'best_results_folds_SVM_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_SVM = f'best_result_SVM_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FOLDS_NAME_DT = f'best_results_folds_DT_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_DT = f'best_result_DT_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FOLDS_NAME_RF = f'best_results_folds_RF_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_RF = f'best_result_RF_{class_ammount}_{cross_validation_type}_{signal_type}.pickle'


PATH_HANDCRAFTED_FEATURE = os.path.join(RESULTS_PATH, HANDCRAFTED_FEATURE_NAME)
PATH_SPECTOGRAM_FEATURE = os.path.join(RESULTS_PATH, SPECTOGRAM_FEATURE_NAME)
PATH_CROSS_VALIDATION = os.path.join(RESULTS_PATH, CROSS_VALIDATION_NAME)
PATH_BEST_RESULTS_FOLDS_SVM = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_SVM)
PATH_BEST_RESULTS_FIXED_PARAMS_SVM = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_SVM)
PATH_BEST_RESULTS_FOLDS_DT = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_DT)
PATH_BEST_RESULTS_FIXED_PARAMS_DT = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_DT)
PATH_BEST_RESULTS_FOLDS_RF = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_RF)
PATH_BEST_RESULTS_FIXED_PARAMS_RF = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_RF)

info_dataframe = create_dataset(signal_type)

info_dataframe = info_dataframe[info_dataframe['label']!= '']
signal_analysis = SignalAnalysis(info_dataframe, signal_type)

with open(PATH_HANDCRAFTED_FEATURE, 'rb') as output:
    features_T = pickle.load(output)
with open(os.path.join(RESULTS_PATH, INFORMATION_NAME), 'rb') as output:
    info_dataframe_T = pickle.load(output)