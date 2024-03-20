import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pickle
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, confusion_matrix, classification_report, mean_squared_error
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import classification_report,f1_score, confusion_matrix,accuracy_score
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, LeaveOneOut, LeaveOneGroupOut, train_test_split, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
"""
FOR TORCH MY RECOMENDATION IS TO USE GOOGLE COLAB TO USE THEIR GPUs, IN CASE
THE LOCAL COMPUTER HAS A GPU AND IS CONFIGURED THEN THE CODE WILL RUN IN GPU MODE
IF NOT THE CODE RUNs IN CPU BUT SLOWER
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
import datetime
from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F
from constant import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def statistical_information(y_true, y_predicted, classes):
  f1 = f1_score(y_true, y_predicted, average='weighted')
  
  print("Weighted F1 Score:", f1)

  conf_matrix = confusion_matrix(y_true, y_predicted, labels = classes)
  print("Class order: ", classes)
  print("Confusion Matrix:")
  print(conf_matrix)

  num_classes = len(classes)  # Number of unique classes in the true labels

  sensitivity = []
  specificity = []

  for i, class_name in enumerate(classes):
      true_positive = conf_matrix[i, i]
      false_positive = sum(conf_matrix[:, i]) - true_positive
      false_negative = sum(conf_matrix[i, :]) - true_positive
      true_negative = sum(sum(conf_matrix)) - true_positive - false_positive - false_negative

      sensitivity_i = true_positive / (true_positive + false_negative)
      specificity_i = true_negative / (true_negative + false_positive)

      sensitivity.append(sensitivity_i)
      specificity.append(specificity_i)

  print("Sensitivity (True Positive Rate) for each class:", sensitivity)
  print("Specificity (True Negative Rate) for each class:", specificity)
  return f1, sensitivity, specificity,conf_matrix

def classical_model_dataset(features, task_number_column, signal_type = 'T', full_classes=False):
    """Function to create a dataset that will be used in classical models

    Args:
        features (DataFrame): Dataframe that contains the features that will be used
        task_number_column (Series): Column from info_dataframe that contains the task number 
        signal_type (str): String with the type of the signal can be either: 'T', 'MT', 'M'. Defaults to 'T'
        full_classes (bool, optional): Use all classes or just S-C-T. Defaults to False.

    Returns:
        DataFrame: Contains all the features 
        Series: Contains the labels of the features
        list: Contains the ID of each feature
    """
    # signal_type = 'throat'
    signal_type = signal_type.lower()
    if signal_type == 'mt':
        signals = ['throat','x','y','z']    
    elif signal_type == 't':
        signals = ['throat','condenser']
    elif signal_type == 'm':
        signals = ['x','y','z']
    else:
        print("Tipo de señal incorrecta, solo se puede elegir entre: 'T', 'MT' y 'M'")
    
    for i, signal in enumerate(signals):
      if signal == 'throat' or signal == 'condenser':
        feature_selection = [
            f'zcr_{signal}',
            f'energy_{signal}_mean',
            f'energy_{signal}_std',
            f'spectral_centroid_{signal}_mean',
            f'spectral_centroid_{signal}_std',
            f'mfccs_{signal}_mean',
            f'delta_mfccs_{signal}_mean',
            f'mfccs_{signal}_std',
            f'delta_mfccs_{signal}_std',
            f'mfccs_{signal}_skewness',
            f'delta_mfccs_{signal}_skewness',
            f'mfccs_{signal}_kurtosis',
            f'delta_mfccs_{signal}_kurtosis',
        ]
        temp = features[feature_selection]
        mean_df = pd.DataFrame(
            temp[f'mfccs_{signal}_mean'].tolist(), index=temp.index)
        mean_df.columns = [f'mfccs_{signal}_mean_{i}' for i in range(
            1, len(mean_df.columns) + 1)]
        delta_mean_df = pd.DataFrame(
            temp[f'delta_mfccs_{signal}_mean'].tolist(), index=temp.index)
        delta_mean_df.columns = [f'delta_mfccs_{signal}_mean_{i}' for i in range(
            1, len(delta_mean_df.columns) + 1)]

        std_df = pd.DataFrame(
            temp[f'mfccs_{signal}_std'].tolist(), index=temp.index)
        std_df.columns = [f'mfccs_{signal}_std_{i}' for i in range(
            1, len(std_df.columns) + 1)]
        delta_std_df = pd.DataFrame(
            temp[f'delta_mfccs_{signal}_std'].tolist(), index=temp.index)
        delta_std_df.columns = [f'delta_mfccs_{signal}_std_{i}' for i in range(
            1, len(delta_std_df.columns) + 1)]

        skew_df = pd.DataFrame(
            temp[f'mfccs_{signal}_skewness'].tolist(), index=temp.index)
        skew_df.columns = [f'mfccs_{signal}_skewness_{i}' for i in range(
            1, len(mean_df.columns) + 1)]
        delta_skew_df = pd.DataFrame(
            temp[f'delta_mfccs_{signal}_skewness'].tolist(), index=temp.index)
        delta_skew_df.columns = [f'delta_mfccs_{signal}_skewness_{i}' for i in range(
            1, len(delta_skew_df.columns) + 1)]

        kurtosis_df = pd.DataFrame(
            temp[f'mfccs_{signal}_kurtosis'].tolist(), index=temp.index)
        kurtosis_df.columns = [f'mfccs_{signal}_kurtosis_{i}' for i in range(
            1, len(kurtosis_df.columns) + 1)]
        delta_kurtosis_df = pd.DataFrame(
            temp[f'delta_mfccs_{signal}_kurtosis'].tolist(), index=temp.index)
        delta_kurtosis_df.columns = [f'delta_mfccs_{signal}_kurtosis_{i}' for i in range(
            1, len(delta_kurtosis_df.columns) + 1)]
        if i == 0:
          basic = features[[f'zcr_{signal}', f'energy_{signal}_mean', f'energy_{signal}_std',
                            f'spectral_centroid_{signal}_mean', f'spectral_centroid_{signal}_std']]
          basic_no_text = features[[f'zcr_{signal}', f'energy_{signal}_mean',
                                    f'energy_{signal}_std', f'spectral_centroid_{signal}_mean', f'spectral_centroid_{signal}_std']]
        else:
          basic = pd.concat([basic, features[[f'zcr_{signal}', f'energy_{signal}_mean', f'energy_{signal}_std',
                                            f'spectral_centroid_{signal}_mean', f'spectral_centroid_{signal}_std']]],axis=1)
        basic = pd.concat([basic, mean_df, std_df, skew_df, kurtosis_df, delta_mean_df,
                        delta_std_df, delta_skew_df, delta_kurtosis_df], axis=1)

      else:

        feature = ['duration',
             f'energy_{signal}_mean',
             f'energy_{signal}_std',
             f'energy_{signal}_skew',
             f'energy_{signal}_kurt',
             f'coefs_{signal}_aur_0',
             f'coefs_{signal}_aur_1',
             f'coefs_{signal}_aur_2',
             f'coefs_{signal}_aur_3',
             f'coefs_{signal}_aur_4',
             f'coefs_{signal}_aur_5',
             f'coefs_{signal}_aur_6',
             f'coefs_{signal}_aur_7',
             f'coefs_{signal}_aur_8',
             f'coefs_{signal}_aur_9',
             f'{signal}_mean',
             f'{signal}_std',
             f'{signal}_skew',
             f'{signal}_kurt']
        if i == 0:
            basic = features[feature]
        else:
            basic = pd.concat([basic, features[feature]],axis=1)

    extracted_col = task_number_column

    X = basic

    if not full_classes:
        y = features['label'].map(lambda x: x[0])
    else:
        y = features['label'].map(lambda x: x.strip())
    group = list(features['ID'])

    return X, y, group

def deep_model_dataset(feature_set, one_window=True):

    list_classes = list(set(y))
    max_lenth = 0
    X = []
    y = []
    group = []
    # WORKING WITH THE WHOLE AUDIO SPECTOGRAM
    if one_window:
        # FIND MAX LENGTH FOR PADDING
        for segment in feature_set.index:

            data_array = feature_set['mel_spectogram'].loc[segment][0]
            if max_lenth < len(data_array):
                max_lenth = len(data_array)
                max_shape = data_array.reshape(128, -1).shape

        for segment in feature_set.index:
            data_array = feature_set['mel_spectogram'].loc[segment]
            label = feature_set['label'].loc[segment]
            ID_number = feature_set['ID'].loc[segment]

            target_height = max_shape[0]
            target_width = max_shape[1]

            data_array = data_array[0]
            data_array = data_array.reshape(128, -1)

            pad_height = max(0, target_height - data_array.shape[0])
            pad_width = max(0, target_width - data_array.shape[1])

            # Calculate padding on both sides of each dimension
            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad

            padded_spectrogram = np.zeros((target_height, target_width))

            # Copy the smaller spectrogram into the center of the larger array
            padded_spectrogram[top_pad:top_pad + data_array.shape[0],
                               left_pad:left_pad + data_array.shape[1]] = data_array
            padded_spectrogram = padded_spectrogram.reshape(-1)
            X = np.vstack([X, padded_spectrogram]) if len(
                X) else padded_spectrogram

            # SELECT THREE OR FIVE CLASSES
            y.append(list_classes.index(label[0]))
            group.append(int(ID_number))

    # WORKING WITH WINDOWS OF THE SPECTOGRAM
    else:
        for segment in feature_set.index:
            data_array = feature_set['mel_spectogram'].loc[segment]
            label = feature_set['label'].loc[segment]
            ID_number = feature_set['ID'].loc[segment]
            for window in data_array:
                X = np.vstack([X, window]) if len(X) else window
                # SELECT THREE OR FIVE CLASSES
                y.append(list_classes.index(label[0]))
                group.append(int(ID_number))

    max_shape = window.reshape(128, -1).shape
    return X, y, group, max_shape

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def cross_validation(path_store_cross_validation,X, y, group, LOSO=True):
    if os.path.exists(path_store_cross_validation):
        print("CrossValidation already exist, no need to create a new one")
        return

    if LOSO:
        loo = LeaveOneGroupOut()
        store_array_train_LOO = []
        store_array_test_LOO = []
        for i, (train_index, test_index) in enumerate(loo.split(X, y, group)):
            print(f"Fold {i}, length train: {len(train_index)}, length test: {len(test_index)}")
            store_array_train_LOO.append(train_index)
            store_array_test_LOO.append(test_index)

        store_dict = {}
        for i, (train, test) in enumerate(zip(store_array_train_LOO, store_array_test_LOO)):
            fold_name = f"Fold {i+1}"
            store_dict[fold_name] = {'train': train, 'test': test}

    else:
        sgkf = StratifiedGroupKFold(n_splits=5)
        store_array_train = []
        store_array_test = []
        for i, (train_index, test_index) in enumerate(sgkf.split(X, y, group)):
            print(f"Fold {i}")

            store_array_train.append(train_index)
            store_array_test.append(test_index)

        store_dict = {}
        for i, (train, test) in enumerate(zip(store_array_train, store_array_test)):
            fold_name = f"Fold {i+1}"
            store_dict[fold_name] = {'train': train, 'test': test}
    with open(path_store_cross_validation, 'wb') as handle:
        pickle.dump(store_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return store_dict


def DecisionTree(X,y,folds, optimizer=True,*args, **kwargs):
    result_dict = {}
    if optimizer:
        param_grid = {'model__criterion':['gini','entropy'],
              'model__max_depth':np.arange(3,19).tolist()[0::2],
              'model__min_samples_split':np.arange(2,11).tolist()[0::2],
              'model__max_leaf_nodes':np.arange(3,20).tolist()[0::2]}
    acc_array = []
    for i, folder in enumerate(folds):
        #Get info of each folder
        train_set_folder = folds[folder]['train']
        test_set_folder = folds[folder]['test']

        X_train = X.iloc[train_set_folder]
        y_train = y.iloc[train_set_folder]

        X_test = X.iloc[test_set_folder]
        y_test = y.iloc[test_set_folder]
        print(f"FOLD #{i+1}")
        print(f"X train shape: {X_train.shape} and y train shape: {y_train.shape}")
        print(f"X test shape: {X_test.shape} and y test shape: {y_test.shape}")
        if optimizer:
            dt_classifier = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', DecisionTreeClassifier())
            ])
            grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            classes_order = grid_search.classes_
            accuracy = best_model.score(X_test, y_test)
            y_predicted = best_model.predict(X_test)
            #decision_scores = best_model.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'best_params':best_params,
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'class_order':classes_order}

        else:
            best_criterion = kwargs.get('criterion')
            best_depth = kwargs.get('max_depth')
            best_samples_split = kwargs.get('min_samples_split')
            best_leaf_nodes = kwargs.get('max_leaf_nodes')
            clf = make_pipeline(StandardScaler(), DecisionTreeClassifier(criterion = best_criterion, max_depth = best_depth, min_samples_split = best_samples_split,max_leaf_nodes = best_leaf_nodes))

            clf.fit(X_train, y_train)
            classes_order = clf.classes_
            y_predicted = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_predicted)

            #decision_scores = clf.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'param_used':{'criterion':best_criterion,
                                                        'max_depth':best_depth,
                                                        'min_samples_split':best_samples_split,
                                                        'max_leaf_nodes':best_leaf_nodes},
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'class_order':classes_order}
        
    return result_dict

def RandomForest(X,y,folds, optimizer=True,*args, **kwargs):
    result_dict = {}
    if optimizer:
        param_grid = {'model__max_features':['log2','sqrt'],
              'model__max_depth':np.arange(10,100).tolist()[0::10],
              'model__min_samples_split':np.arange(2,11).tolist()[0::2],
              'model__n_estimators':np.arange(200,2100).tolist()[0::2000]}
    acc_array = []
    for i, folder in enumerate(folds):
        train_set_folder = folds[folder]['train']
        test_set_folder = folds[folder]['test']

        X_train = X.iloc[train_set_folder]
        y_train = y.iloc[train_set_folder]

        X_test = X.iloc[test_set_folder]
        y_test = y.iloc[test_set_folder]
        print(f"FOLD #{i+1}")
        print(f"X train shape: {X_train.shape} and y train shape: {y_train.shape}")
        print(f"X test shape: {X_test.shape} and y test shape: {y_test.shape}")
        if optimizer:
            dt_classifier = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier())
            ])
            grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            classes_order = grid_search.classes_
            accuracy = best_model.score(X_test, y_test)
            y_predicted = best_model.predict(X_test)
            #decision_scores = best_model.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'best_params':best_params,
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'class_order':classes_order}

        else:
            best_max_features = kwargs.get('max_features')
            best_depth = kwargs.get('max_depth')
            best_samples_split = kwargs.get('min_samples_split')
            best_n_estimators = kwargs.get('n_estimators')
            clf = make_pipeline(StandardScaler(), RandomForestClassifier(max_features = best_max_features, max_depth = best_depth, min_samples_split = best_samples_split,n_estimators = best_n_estimators))

            clf.fit(X_train, y_train)
            classes_order = clf.classes_
            y_predicted = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_predicted)

            #decision_scores = clf.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'param_used':{'max_features':best_max_features,
                                                        'max_depth':best_depth,
                                                        'min_samples_split':best_samples_split,
                                                        'n_estimators':best_n_estimators},
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'class_order':classes_order}
        
    return result_dict


def SVM_Classifier(X,y,folds,optimizer = True,*args, **kwargs):
    result_dict = {}
    if optimizer:
        param_grid = {
            'model__C': [0.00001,0.0005,0.0001,0.005,0.001, 0.01, 0.1, 1],
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': [0.00001,0.0005,0.0001,0.005,0.001,0.001, 0.01, 0.1, 1],
        }
    feature_name = list(X.columns)
    print(feature_name)
    for i, folder in enumerate(folds):
        #Get info of each folder
        train_set_folder = folds[folder]['train']
        test_set_folder = folds[folder]['test']

        X_train = X.iloc[train_set_folder]
        y_train = y.iloc[train_set_folder]

        X_test = X.iloc[test_set_folder]
        y_test = y.iloc[test_set_folder]

        print(f"FOLD #{i+1}")
        print(f"X train shape: {X_train.shape} and y train shape: {y_train.shape}")
        print(f"X test shape: {X_test.shape} and y test shape: {y_test.shape}")
        if optimizer:
            svm_classifier = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', SVC(probability=True))
            ])
            grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, scoring='accuracy')
# =============================================================================
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
# =============================================================================
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            classes_order = grid_search.classes_
            accuracy = best_model.score(X_test, y_test)
            y_predicted = best_model.predict(X_test)
            decision_scores = best_model.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'best_params':best_params,
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'scores':decision_scores,
                                            'class_order':classes_order}
        else:
            best_C = kwargs.get('C')
            best_gamma = kwargs.get('gamma')
            best_kernel = kwargs.get('kernel')
            clf = make_pipeline(StandardScaler(), SVC(C = best_C,gamma = best_gamma,kernel = best_kernel,probability=True))

            clf.fit(X_train, y_train)
            classes_order = clf.classes_
            y_predicted = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_predicted)

            decision_scores = clf.decision_function(X_test)
            f1, sensitivity, specificity,c_matrix = statistical_information(list(y_test),list(y_predicted),classes_order)
            print(accuracy)

            result_dict[f'Fold {i+1}'] = {'param_used':{'C':best_C,
                                                        'gamma':best_gamma,
                                                        'kernel':best_kernel},
                                            'accuracy_best_model':accuracy,
                                            'f1_score':f1,
                                            'sensitivity':sensitivity,
                                            'specificity':specificity,
                                            'confusion_matrix':c_matrix,
                                            'scores':decision_scores,
                                            'class_order':classes_order}

    return result_dict
    
def SVM_Optimization(X,y, cv_folds, PATH_BEST_RESULTS_FOLDS,PATH_BEST_RESULTS_FIXED_PARAMS):
    result = SVM_Classifier(X,y,cv_folds)
    with open(PATH_BEST_RESULTS_FOLDS, 'wb') as handle:
        pickle.dump(result, handle)
    C_array = []
    gamma_array = []
    kernel_array = []
    for key in result:
        C_array.append(result[key]['best_params']['model__C'])
        gamma_array.append(result[key]['best_params']['model__gamma'])
        kernel_array.append(result[key]['best_params']['model__kernel'])


    best_C = max(set(C_array), key=C_array.count)
    best_gamma = max(set(gamma_array), key=gamma_array.count)
    best_kernel = max(set(kernel_array), key=kernel_array.count)
    best_params_dict = {'C':best_C,'gamma':best_gamma,'kernel':best_kernel}
    result_best_param = SVM_Classifier(X,y,cv_folds,False,**best_params_dict)
    with open(PATH_BEST_RESULTS_FIXED_PARAMS, 'wb') as handle:
        pickle.dump(result_best_param, handle)
    return result_best_param
def DT_Optimization(X,y, cv_folds, PATH_BEST_RESULTS_FOLDS,PATH_BEST_RESULTS_FIXED_PARAMS):
    result = DecisionTree(X,y,cv_folds)
    with open(PATH_BEST_RESULTS_FOLDS, 'wb') as handle:
        pickle.dump(result, handle)
    criterion_array = []
    depth_array = []
    samples_split_array = []
    leaf_nodes_array = []
    for key in result:
        criterion_array.append(result[key]['best_params']['model__criterion'])
        depth_array.append(result[key]['best_params']['model__max_depth'])
        samples_split_array.append(result[key]['best_params']['model__min_samples_split'])
        leaf_nodes_array.append(result[key]['best_params']['model__max_leaf_nodes'])


    best_criterion = max(set(criterion_array), key=criterion_array.count)
    best_depth = max(set(depth_array), key=depth_array.count)
    best_samples_split = max(set(samples_split_array), key=samples_split_array.count)
    best_leaf_nodes = max(set(leaf_nodes_array), key=leaf_nodes_array.count)
    best_params_dict = {'criterion':best_criterion,'max_depth':best_depth,'min_samples_split':best_samples_split, 'max_leaf_nodes':best_leaf_nodes}
    result_best_param = DecisionTree(X,y,cv_folds,False,**best_params_dict)
    with open(PATH_BEST_RESULTS_FIXED_PARAMS, 'wb') as handle:
        pickle.dump(result_best_param, handle)
    return result_best_param

def RF_Optimization(X,y, cv_folds, PATH_BEST_RESULTS_FOLDS,PATH_BEST_RESULTS_FIXED_PARAMS):
    result = RandomForest(X,y,cv_folds)
    with open(PATH_BEST_RESULTS_FOLDS, 'wb') as handle:
        pickle.dump(result, handle)
    max_features_array = []
    depth_array = []
    samples_split_array = []
    n_estimators_array = []
    
    for key in result:
        max_features_array.append(result[key]['best_params']['model__max_features'])
        depth_array.append(result[key]['best_params']['model__max_depth'])
        samples_split_array.append(result[key]['best_params']['model__min_samples_split'])
        n_estimators_array.append(result[key]['best_params']['model__n_estimators'])


    best_max_features = max(set(max_features_array), key=max_features_array.count)
    best_depth = max(set(depth_array), key=depth_array.count)
    best_samples_split = max(set(samples_split_array), key=samples_split_array.count)
    best_n_estimators = max(set(n_estimators_array), key=n_estimators_array.count)
    best_params_dict = {'max_features':best_max_features,'max_depth':best_depth,'min_samples_split':best_samples_split, 'n_estimators':best_n_estimators}
    result_best_param = RandomForest(X,y,cv_folds,False,**best_params_dict)
    with open(PATH_BEST_RESULTS_FIXED_PARAMS, 'wb') as handle:
        pickle.dump(result_best_param, handle)
    return result_best_param
class CNNModel(nn.Module):
    def __init__(self, n_mel, linear_size=128, input_channel_1=16):
        super(CNNModel, self).__init__()
        # input shape [1,128,49]
        self.n_mel = n_mel
        self.input_channel_1 = input_channel_1

        self.linear_size = linear_size

        # [16, 128, 49]
        self.conv1 = nn.Conv2d(1, self.input_channel_1, kernel_size=(
            3, 3), stride=1, padding=1, bias=False)
        # self.batch = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        # [16, 64, 24]
        self.conv2 = nn.Conv2d(self.input_channel_1, 16, kernel_size=(
            3, 3), stride=1, padding=1, bias=False)

        self.flatten_func = nn.Flatten()
        self.fc1 = nn.LazyLinear(self.linear_size)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.linear_size, 64)

        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten_func(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class EarlyStopping():
    def __init__(self, tolerance=10, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class ConvolutionalNeuralNet():
    def __init__(self, network, lr, weight_decay):
        self.network = network.to(device)
        # , weight_decay=weight_decay
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.globaliter = 0

    def train(self, loss_function, epochs, batch_size,
              train_dataset, test_dataset):

        #  creating log
        log_dict = {
            'training_loss_per_batch': [],
            'validation_loss_per_batch': [],
            'training_accuracy_per_epoch': [],
            'validation_accuracy_per_epoch': []
        }

        def accuracy(network, dataloader, binary=False):
            network.eval()
            total_correct = 0
            total_instances = 0
            for images, labels in tqdm(dataloader):
                images, labels = images.to(device), labels.to(device)

                if binary:
                    predictions = (network(images) >= 0.5).float()
                    predictions = predictions.reshape(-1)
                else:
                    predictions = torch.argmax(network(images), dim=1)

                correct_predictions = sum(predictions == labels).item()

                total_correct += correct_predictions
                total_instances += len(images)
            return round(total_correct/total_instances, 3)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)
        early_stopping = EarlyStopping(tolerance=5, min_delta=0.2)
        self.network.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            train_losses = []
            train_loss_mean = 0
            count = 0
            self.globaliter += 1
            for i, (spectogram, labels) in enumerate(train_loader):
                spectogram = spectogram.to(device)
                labels = labels.to(device)
                #  resetting gradients
                self.optimizer.zero_grad()
                #  making predictions
                predictions = self.network(spectogram)
                predictions = predictions.to(device)
                # predictions = predictions.reshape(-1)
                #  computing loss
                labels = labels.type(torch.LongTensor)

                # BINARY
                # labels = labels.unsqueeze(1)
                # labels = labels.float()

                labels = labels.to(device)
                loss = loss_function(predictions, labels)
                train_loss_mean += loss
                count += 1
                log_dict['training_loss_per_batch'].append(loss.item())
                train_losses.append(loss.item())
                # if (i+1)%10==0:
                # print(f'Epoch: {epoch+1} / {epochs}, step = {i+1}/{n_total_step}, loss = {loss.item():.4f}')
                #  computing gradients
                loss.backward()
                #  updating weights
                self.optimizer.step()
            with torch.no_grad():
                print('deriving training accuracy...')
                #  computing training accuracy
                train_accuracy = accuracy(self.network, train_loader)
                log_dict['training_accuracy_per_epoch'].append(train_accuracy)
                train_loss_mean /= count
            # with train_summary_writer.as_default():
            #     summary.scalar('loss', train_loss_mean, step=self.globaliter)
            #     summary.scalar('accuracy', train_accuracy, step=self.globaliter)
            #  validation
            print('validating...')
            val_losses = []
            self.network.eval()

            with torch.no_grad():
                val_loss_mean = 0
                count = 0
                for images, labels in tqdm(test_loader):
                    #  sending data to device
                    images, labels = images.to(device), labels.to(device)
                    #  making predictions
                    predictions = self.network(images)
                    predictions = predictions.to(device)
                    #  computing loss
                    labels = labels.type(torch.LongTensor)

                    # BINARY
                    # labels = labels.unsqueeze(1)
                    # labels = labels.float()

                    labels = labels.to(device)
                    val_loss = loss_function(predictions, labels)
                    val_loss_mean += val_loss
                    count += 1
                    log_dict['validation_loss_per_batch'].append(
                        val_loss.item())
                    val_losses.append(val_loss.item())
                #  computing accuracy
                print('deriving validation accuracy...')
                val_accuracy = accuracy(self.network, test_loader)
                log_dict['validation_accuracy_per_epoch'].append(val_accuracy)
                val_loss_mean /= count

            train_losses = np.array(train_losses).mean()
            val_losses = np.array(val_losses).mean()

            print(f'training_loss: {round(train_losses, 4)}  training_accuracy: ' +
                  f'{train_accuracy}  validation_loss: {round(val_losses, 4)} ' +
                  f'validation_accuracy: {val_accuracy}\n')

            early_stopping(train_losses, val_losses)
            if early_stopping.early_stop:
                print("We are at epoch:", i)
                break
        return log_dict, self.network, self.optimizer

    def predict(self, x):
        return self.network(x)


def training_step(dict_cross_validation,path_store_results,X,y, max_shape, batch=32):
    results = {}

    for i, folder in enumerate(dict_cross_validation):
        print("FOLDER " + str(i))
        print("***************"*10)
        # Get info of each folder
        train_set_folder = dict_cross_validation[folder]['train']
        test_set_folder = dict_cross_validation[folder]['test']
        # print(train_set)
        input_train_filtered = torch.from_numpy(
            X[train_set_folder]).to(torch.float32)
        input_train_filtered = input_train_filtered.reshape(
            -1, 1, 128, max_shape[1])
        label_train_filtered = torch.from_numpy(
            np.array(y)[train_set_folder]).to(torch.float32)

        input_test_filtered = torch.from_numpy(
            X[test_set_folder]).to(torch.float32)
        input_test_filtered = input_test_filtered.reshape(
            -1, 1, 128, max_shape[1])
        label_test_filtered = torch.from_numpy(
            np.array(y)[test_set_folder]).to(torch.float32)

        train_dataset = torch.utils.data.TensorDataset(
            input_train_filtered, label_train_filtered)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch, shuffle=True)
        batch_test = 32
        test_dataset = torch.utils.data.TensorDataset(
            input_test_filtered, label_test_filtered)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_test)
        # lr_array = [1e-4, 1e-3, 1e-2]
        # weight_decay = [1e-6, 1e-5, 1e-4]
        lr_value = [1e-3]
        weight_decay_value = [1e-5]
        linear_size_array = [64, 128, 256]
        input_channel = [16, 32, 64]
        iterable = np.array(np.meshgrid(
            lr_value, weight_decay_value, linear_size_array, input_channel)).T.reshape(-1, 4)
        key_name_general = "Fold "+str(i)
        results[key_name_general] = {}
        for lr, weight_decay, l_size, i_channel in iterable:
            modelClass = ConvolutionalNeuralNet(
                CNNModel(128, int(l_size), int(i_channel)), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()

            epochs = 100
            batch_size = batch
            log_dict, model_trained, optimizer_trained = modelClass.train(
                criterion, epochs, batch_size, train_dataset, test_dataset)
            key_name = 'Lr= ' + str(lr) + ' wd= ' + str(weight_decay) + \
                'linear_size= ' + str(l_size) + \
                ' input_channel= ' + str(i_channel)
            results[key_name_general][key_name] = log_dict

    with open(os.path.join(RESULTS_PATH, path_store_results), 'wb') as handle:
        pickle.dump(results, handle)