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

signal_type = 'MT'
unique_signal_type_fusion = 'M'
full_classes = True
LOSO = True

class_ammount = 'five' if full_classes else 'three'
cross_validation_type = 'LOSO' if LOSO else 'SGKFold'

INFORMATION_NAME = f'information_dataset_{signal_type}_{unique_signal_type_fusion}.pickle'
HANDCRAFTED_FEATURE_NAME = f'features_dataset_{signal_type}_{unique_signal_type_fusion}.pickle'
SPECTOGRAM_FEATURE_NAME = f'spectograms_dataset_{signal_type}_{unique_signal_type_fusion}.pickle'
CROSS_VALIDATION_NAME = f'folds_{class_ammount}_{cross_validation_type}_{signal_type}_{unique_signal_type_fusion}.pickle'
BEST_RESULTS_FOLDS_NAME_SVM = f'best_results_folds_SVM_{class_ammount}_{cross_validation_type}_{signal_type}_{unique_signal_type_fusion}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_SVM = f'best_result_SVM_{class_ammount}_{cross_validation_type}_{signal_type}_{unique_signal_type_fusion}.pickle'
BEST_RESULTS_FOLDS_NAME_DT = f'best_results_folds_DT_{class_ammount}_{cross_validation_type}_{signal_type}_{unique_signal_type_fusion}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_DT = f'best_result_DT_{class_ammount}_{cross_validation_type}_{signal_type}_{unique_signal_type_fusion}.pickle'
BEST_RESULTS_FOLDS_NAME_RF = f'best_results_folds_RF_{class_ammount}_{cross_validation_type}_{signal_type}_{unique_signal_type_fusion}.pickle'
BEST_RESULTS_FIXED_PARAMS_NAME_RF = f'best_result_RF_{class_ammount}_{cross_validation_type}_{signal_type}_{unique_signal_type_fusion}.pickle'

PATH_INFORMATION = os.path.join(RESULTS_PATH, INFORMATION_NAME)
PATH_HANDCRAFTED_FEATURE = os.path.join(RESULTS_PATH, HANDCRAFTED_FEATURE_NAME)
PATH_SPECTOGRAM_FEATURE = os.path.join(RESULTS_PATH, SPECTOGRAM_FEATURE_NAME)
PATH_CROSS_VALIDATION = os.path.join(RESULTS_PATH, CROSS_VALIDATION_NAME)
PATH_BEST_RESULTS_FOLDS_SVM = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_SVM)
PATH_BEST_RESULTS_FIXED_PARAMS_SVM = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_SVM)
PATH_BEST_RESULTS_FOLDS_DT = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_DT)
PATH_BEST_RESULTS_FIXED_PARAMS_DT = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_DT)
PATH_BEST_RESULTS_FOLDS_RF = os.path.join(RESULTS_PATH, BEST_RESULTS_FOLDS_NAME_RF)
PATH_BEST_RESULTS_FIXED_PARAMS_RF = os.path.join(RESULTS_PATH, BEST_RESULTS_FIXED_PARAMS_NAME_RF)

info_dataframe = create_dataset(PATH_INFORMATION,signal_type)

info_dataframe = info_dataframe[info_dataframe['label']!= '']

signal_analysis = SignalAnalysis(info_dataframe, unique_signal_type_fusion)
signal_analysis.store_handcrafeted_features(PATH_HANDCRAFTED_FEATURE)
with open(PATH_HANDCRAFTED_FEATURE, 'rb') as output:
    features = pickle.load(output)
with open(PATH_INFORMATION, 'rb') as output:
    info_dataframe = pickle.load(output)
task_number = info_dataframe["task_number"]

X,y, group = classical_model_dataset(features,task_number, unique_signal_type_fusion,full_classes)
cross_validation(PATH_CROSS_VALIDATION, X, y, group, LOSO=LOSO)
with open(PATH_CROSS_VALIDATION, 'rb') as output:
    cv_folds = pickle.load(output)

result_best_param_SVM = SVM_Optimization(X,y,cv_folds,PATH_BEST_RESULTS_FOLDS_SVM,PATH_BEST_RESULTS_FIXED_PARAMS_SVM)
result_best_param_DT = DT_Optimization(X,y,cv_folds,PATH_BEST_RESULTS_FOLDS_DT,PATH_BEST_RESULTS_FIXED_PARAMS_DT)
result_best_param_RF = RF_Optimization(X,y,cv_folds,PATH_BEST_RESULTS_FOLDS_RF,PATH_BEST_RESULTS_FIXED_PARAMS_RF)

    
#%%
result_to_check = result_best_param_SVM
accuracy_toal = []
f1_score_total = []

for key in result_to_check.keys():
  accuracy_toal.append(result_to_check[key]["accuracy_best_model"])
  f1_score_total.append(result_to_check[key]["f1_score"])

mean_accuracy = np.mean(accuracy_toal)
mean_f1_score = np.mean(f1_score_total)

std_accuracy = np.std(accuracy_toal)
std_f1_score = np.std(f1_score_total)

print(mean_accuracy, "+-", std_accuracy)
print(mean_f1_score, "+-", std_f1_score)

