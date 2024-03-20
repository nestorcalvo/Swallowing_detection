import pandas as pd
import numpy as np
import librosa
from scipy.stats import skew, norm, kurtosis
from scipy.signal import hilbert
from scipy.interpolate import CubicSpline
from utils import calculate_energy, spectogram_calc, butter_lowpass_filter
from statsmodels.tsa.ar_model import AutoReg

def handcrafted_feature_creation(dataset, hop_length, frame_length, signal_type):

    data = pd.DataFrame()
    if signal_type == 'mt':
        signals = ['throat','x','y','z']
    elif signal_type == 't':
        signals = ['throat','condenser']
    elif signal_type == 'm':
        signals = ['x','y','z']
    else:
        print("Tipo de señal incorrecta, solo se puede elegir entre: 'T', 'MT' y 'M'")
    
    data[['ID', 'label', 'time_start', 'time_finish', 'time_array', 'fs']] = dataset[[
        'ID', 'label', 'time_start', 'time_finish', 'time_array', 'fs']]
    dataset = dataset[dataset['label']!= '']
    data['duration'] = data['time_finish'] - data['time_start']
    
    fs = data['fs']
    for signal in signals:
        if signal == 'throat' or signal == 'condenser':
            # ZERO-CROSS RATING
            data[f'zcr_{signal}'] = dataset[f'{signal}_segment'].map(
                lambda x: sum(librosa.zero_crossings(x)))
            # ENERGY
            data[f'energy_{signal}'] = dataset[f'{signal}_segment'].apply(
                lambda x: calculate_energy(x, frame_length, hop_length))
        
            data[f'energy_{signal}_mean'] = data[f'energy_{signal}'].apply(
                lambda energy_array: np.mean(energy_array))
            
            data[f'energy_{signal}_std'] = data[f'energy_{signal}'].apply(
                lambda energy_array: np.std(energy_array))
            
            # SPECTRAL CENTROID
            data[f'spectral_centroid_{signal}'] = dataset.apply(lambda row: librosa.feature.spectral_centroid(
                y=row[f'{signal}_segment'], sr=row['fs']), axis=1)
            
            data[f'spectral_centroid_{signal}_mean'] = data[f'spectral_centroid_{signal}'].apply(
                lambda centroid: np.mean(centroid))
            
            data[f'spectral_centroid_{signal}_std'] = data[f'spectral_centroid_{signal}'].apply(
                lambda centroid: np.std(centroid))
            
            # MFCCs
            data[f'mfccs_{signal}'] = dataset.apply(lambda row: librosa.feature.mfcc(
                y=row[f'{signal}_segment'], sr=row['fs'], n_mfcc=12), axis=1)
            
            data[f'delta_mfccs_{signal}'] = data[f'mfccs_{signal}'].apply(
                lambda row: librosa.feature.delta(row, order=1))
            
            data[f'mfccs_{signal}_mean'] = data[f'mfccs_{signal}'].apply(
                lambda mfccs: np.mean(mfccs, axis=1))
            
            data[f'delta_mfccs_{signal}_mean'] = data[f'delta_mfccs_{signal}'].apply(
                lambda mfccs: np.mean(mfccs, axis=1))
    
            data[f'mfccs_{signal}_std'] = data[f'mfccs_{signal}'].apply(
                lambda mfccs: np.std(mfccs, axis=1))
            
            data[f'delta_mfccs_{signal}_std'] = data[f'delta_mfccs_{signal}'].apply(
                lambda mfccs: np.std(mfccs, axis=1))
            
            data[f'mfccs_{signal}_skewness'] = data[f'mfccs_{signal}'].apply(
                lambda mfccs: skew(mfccs, axis=1))
            
            data[f'delta_mfccs_{signal}_skewness'] = data[f'delta_mfccs_{signal}'].apply(
                lambda mfccs: skew(mfccs, axis=1))
            
            data[f'mfccs_{signal}_kurtosis'] = data[f'mfccs_{signal}'].apply(
                lambda mfccs: kurtosis(mfccs, axis=1))
            
            data[f'delta_mfccs_{signal}_kurtosis'] = data[f'delta_mfccs_{signal}'].apply(
                lambda mfccs: kurtosis(mfccs, axis=1))
        else:
          fc = 40
          filter_order = 6
          # Hilbert and lowpass filter
          signal_name = f'{signal}_segment'
          
          dataset.loc[:, signal_name] = dataset.loc[:, signal_name].apply(
              lambda x: np.abs(hilbert(x)))

          dataset.loc[:, signal_name] = dataset.apply(
              lambda row: butter_lowpass_filter(row[signal_name], fc, row['fs'], filter_order), axis=1)
          # First 220 ms removed because of peak after hilbert transform
          data.loc[:, signal_name] = dataset.loc[:, signal_name].apply(
              lambda x: x[1200:-1200])
          # Energy
          data[f'{signal}_log'] = data.loc[:, signal_name].apply(
              lambda x: np.log(x))
          data[f'energy_{signal}'] = data[signal_name].apply(
              lambda x: calculate_energy(x, frame_length, hop_length))
          data[f'energy_{signal}_mean'] = data[f'energy_{signal}'].apply(
              lambda energy_array: np.mean(energy_array))
          data[f'energy_{signal}_std'] = data[f'energy_{signal}'].apply(
              lambda energy_array: np.std(energy_array))
          data[f'energy_{signal}_skew'] = data[f'energy_{signal}'].apply(
              lambda energy_array: skew(energy_array))
          data[f'energy_{signal}_kurt'] = data[f'energy_{signal}'].apply(
              lambda energy_array: kurtosis(energy_array))
          new_t = dataset['time_array'].apply(
              lambda x: x[1200:-1200])
          data['new_time_array'] = new_t

          lag = 10
          data[f'coefs_{signal}_aur'] = data[signal_name].apply(lambda x: AutoReg(x, lags = lag).fit().params)
          for i in range(lag):
            data[f'coefs_{signal}_aur_{i}'] = data[f'coefs_{signal}_aur'].apply(lambda x: x[i])
            
            
          data[f'{signal}_normalize'] = data[signal_name].apply(lambda x: (x - np.min(x))/np.max(x))
          data[f'{signal}_mean'] = data[f'{signal}_normalize'].apply(lambda x: np.mean(x))
          data[f'{signal}_std'] = data[f'{signal}_normalize'].apply(lambda x: np.std(x))
          data[f'{signal}_skew'] = data[f'{signal}_normalize'].apply(lambda x: skew(x))
          data[f'{signal}_kurt'] = data[f'{signal}_normalize'].apply(lambda x: kurtosis(x))
          
          data[f'{signal}_gradient'] = data[f'{signal}_normalize'].apply(lambda x: np.gradient(x))
          data.drop('new_time_array', axis = 1, inplace = True)
    return data

def spectograms_feature_creation(dataset, hop_length, frame_length, MELS, signal_type):
    if signal_type == 'mt':
        signals = ['throat','x','y','z']
    elif signal_type == 't':
        signals = ['throat','condenser']
    elif signal_type == 'm':
        signals = ['x','y','z']
    else:
        print("Tipo de señal incorrecta, solo se puede elegir entre: 'T', 'MT' y 'M'")

    data = pd.DataFrame()
    data[['ID', 'label', 'time_start', 'time_finish', 'time_array', 'fs']] = dataset[[
        'ID', 'label', 'time_start', 'time_finish', 'time_array', 'fs']]
    data['duration'] = data['time_finish'] - data['time_start']
    fs = data['fs']
    for signal in signals:
        data[f'mel_spectogram_{signal}'] = dataset.apply(lambda audio_data: spectogram_calc(
            audio_data[f'{signal}_segment'], audio_data['fs'], frame_length, MELS, hop_length, signal), axis=1)
    return data