#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:12:13 2024

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
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz, filtfilt

from constant import *

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def calculate_energy(x, frame_length, hop_length):
      energy = np.array([sum(abs(x[i:i+frame_length]**2)) for i in range(0, len(x), hop_length)])
      return energy  
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

#%%
fc = 40
filter_order = 6
segment = 6
signal = info_dataframe.iloc[segment]['x_segment']
time_segment = info_dataframe.iloc[segment]['time_array']
fs = info_dataframe.iloc[segment]['fs']
hilbert_signal = np.abs(hilbert(signal))
filtered_signal = butter_lowpass_filter(hilbert_signal, fc, fs, filter_order)
coef_variacion_signal = np.std(filtered_signal)/np.mean(filtered_signal)
energy_from_signal = calculate_energy(filtered_signal, 512, 128)

mean_energy = np.mean(energy_from_signal)
std_energy = np.std(energy_from_signal)
coef_variacion_energy = std_energy / mean_energy

histograma = np.histogram(filtered_signal, bins=10)
pdf = np.histogram(filtered_signal, bins=10, density=True)
transformada_fourier = np.fft.fft(filtered_signal)

aceleracion_cambio = np.gradient(filtered_signal)
mean_aceleration = np.mean(aceleracion_cambio)
std_aceleration = np.std(aceleracion_cambio)
skew_aceleration = pd.Series(aceleracion_cambio).skew()
krt_aceleration = pd.Series(aceleracion_cambio).kurtosis()

coef = np.polyfit(time_segment,filtered_signal, fs, 6)

class_name = info_dataframe.iloc[segment]['label']
plt.title("Class type: " + class_name)
plt.plot(filtered_signal)
#plt.plot(hilbert_signal)
plt.show()