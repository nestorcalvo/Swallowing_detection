
# -*- coding: utf-8 -*-
"""
Utils module

This module contains multiple functions that are used during the project, this functions are mainly used to 
preprocess, extract or display the signals that are going to be analyzed


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import glob
import librosa
import librosa.display
import IPython.display as ipd
from scipy.io.wavfile import read
import pickle
import matplotlib.ticker as ticker
from time import sleep
from tqdm import tqdm
from scipy.signal import hilbert
from scipy.signal import butter, lfilter, freqz, filtfilt
import re
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


def extract_window(signal, size, fs):
    """
    Function to extract windows from a signal

    Args:
        signal (int): Signal to be analyzed
        size (int): Length of the window that needs to be extracted. This value
            should be less than len(signal).
        fs (int): Sample frequency of the signal.

    Returns:
        np.array: Returns a numpy array with each each window extracted
    """
    step=int(size*fs)
    n_seg = int(len(signal) / step)
    windows = [signal[i * step : i * step + step] for i in range(n_seg)]
    return np.vstack(windows)

def calculate_energy(x, frame_length, hop_length):
    energy = np.array([sum(abs(x[i:i+frame_length]**2)) for i in range(0, len(x), hop_length)])
    return energy


def spectogram_calc(data_audio,FS,NFFT,MELS,NOVERLAP,signal):
    """
    Function to calculate the spetograms out of the signal

    Args:
        data_audio (np.array): Contains the audio signal 
        FS (int): Sample frequency of the signal
        NFFT (int): Length of the FFT window.
        MELS (int): Number of MEL bands that wants to be generated
        NOVERLAP (int): Number of samples between consecutive frames
        signal (str): Signal type
    Returns:
        list: Returns a list that has all the spectograms appended from one audio
    """

        
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    data_audio=data_audio-np.mean(data_audio)
    count = 1
    if signal == 'x' or signal == 'y' or signal == 'z':
        fc = 40
        filter_order = 6
        x = data_audio
        data_audio = np.abs(hilbert(data_audio))
        data_audio = butter_lowpass_filter(data_audio, fc, FS, filter_order)

    #ventanas=extract_window(data_audio,0.05,FS)
    m_partial = []

    #for i in tqdm(range(len(ventanas))):
    S=librosa.feature.melspectrogram(y=data_audio, sr=FS,
                        S=None, n_fft=NFFT,n_mels=MELS,
                        hop_length=NOVERLAP, power= 1)

    input_cnn = librosa.power_to_db(S, ref=np.max)
    input_cnn = np.array(input_cnn).flatten()
    m_partial.append(input_cnn)
    return m_partial

def read_audio_info(RECORDINGS_PATH, signal_type = 'T'):
    signal_type = signal_type.lower()
    if signal_type == 't':
        pattern = '_t'
        other_signal_pattern = '_m'
    elif signal_type == 'm':
        pattern = '_m'
        other_signal_pattern = '_t'
    else:
        print('Early fusion modality')
    reg_path = f'/**/*_{signal_type}/*.txt'
    files = glob.glob(RECORDINGS_PATH +reg_path, recursive = True)
    info_array = []
    index = 1
    # Find the folder name of a task for a type of signal
    regex_instance = f'[0-9]*_[a-zA-Z]*_{signal_type}'
    p = re.compile(regex_instance)
    print("Creating dataset with segments...")
    for f in files:
        # Gets the name of the folder
        result = p.search(f)
        match = result.group(0)
        # Replace the signal type with the other signal and creates a path
        match_changed = match.replace(pattern, other_signal_pattern)
        other_modality_path = f.replace(match, match_changed)
        # Checks if this new path is there to see if that task has the other signal
        if not os.path.isfile(other_modality_path):
            continue
        # BASIC INFORMATION FOR DATASET
        f = os.path.normpath(f)
        subject_id = int(f.split(os.sep)[-3])

        audio_path = os.path.dirname(os.path.abspath(f))
        if signal_type == 't':
            throat_mic_path = os.path.join(audio_path, 'throat_mic.wav')
            condenser_mic_path = os.path.join(audio_path, 'condenser_mic.wav')
            
            fs, _ = read(throat_mic_path)
            data_audio_throat, fs_throat  = librosa.load(throat_mic_path, sr = fs)
            data_audio_condenser, fs_condenser = librosa.load(condenser_mic_path, sr = fs)
            
        elif signal_type == 'mt':
            throat_mic_path = os.path.join(audio_path, 'throat_mic.wav')
            fs, _ = read(throat_mic_path)
            data_audio_throat, fs_throat  = librosa.load(throat_mic_path, sr = fs)
            
            x_axis_path = os.path.join(audio_path, 'x_axis.wav')
            y_axis_path = os.path.join(audio_path, 'y_axis.wav')
            z_axis_path = os.path.join(audio_path, 'z_axis.wav')
            fs, _ = read(x_axis_path)
            data_x_axis, fs_x_axis  = librosa.load(x_axis_path, sr = fs)
            data_y_axis, fs_y_axis  = librosa.load(y_axis_path, sr = fs)
            data_z_axis, fs_z_axis  = librosa.load(z_axis_path, sr = fs)
            
        elif signal_type == 'm':
            
            condenser_mic_path = os.path.join(audio_path, 'condenser_mic.wav')
            fs, _ = read(condenser_mic_path)
            data_audio_condenser, fs_condenser  = librosa.load(condenser_mic_path, sr = fs)
            
            x_axis_path = os.path.join(audio_path, 'x_axis.wav')
            y_axis_path = os.path.join(audio_path, 'y_axis.wav')
            z_axis_path = os.path.join(audio_path, 'z_axis.wav')
            fs, _ = read(x_axis_path)
            data_x_axis, fs_x_axis  = librosa.load(x_axis_path, sr = fs)
            data_y_axis, fs_y_axis  = librosa.load(y_axis_path, sr = fs)
            data_z_axis, fs_z_axis  = librosa.load(z_axis_path, sr = fs)
        else:
            print("Please choose a correct signal type, either: T, MT, M") 
            return
        camera_path = os.path.join(audio_path, 'camera_0.mp4')
        label_path = f

        
        # OPEN LABEL FILE
        text = open(f, "r")
        for line in text:
            # GET INFORMATION FROM LABEL FILE
            time_start, time_finish, label = line.split("\t")
            time_start = float(time_start)
            time_finish = float(time_finish)

            #GET INDEX OF START AND STOP BASED ON SAMPLE FREQUENCY
            if signal_type == 't':
                start_index_throat = time_start * fs_throat
                start_index_condenser = time_start * fs_condenser

                stop_index_throat = time_finish * fs_throat
                stop_index_condenser = time_finish * fs_condenser

                time_array = np.arange(time_start,time_finish,1/fs)
                # AUDIO SEGMENTS FOR BOTH SIGNALS
                data_audio_throat_segment = data_audio_throat[int(start_index_throat):int(stop_index_throat)+1]
                data_audio_condenser_segment = data_audio_condenser[int(start_index_condenser):int(stop_index_condenser)+1]

                diference = len(time_array)-len(data_audio_throat_segment)+1


                data_audio_throat_segment = data_audio_throat[int(start_index_throat):int(stop_index_throat)+diference]
                data_audio_condenser_segment = data_audio_condenser[int(start_index_condenser):int(stop_index_condenser)+diference]


                task_number = str(audio_path.split('/')[-1].split('_')[0])
                label = label.replace('\n','')
                fs = fs_throat
                # STORE INFORMATION IN ARRAY TO TRANSFORM IT INTO A DATAFRAME
                info = [index,
                        subject_id,
                        time_start,
                        time_finish,
                        time_array,
                        fs,
                        data_audio_throat_segment,
                        data_audio_condenser_segment,
                        label,
                        audio_path,
                        throat_mic_path,
                        condenser_mic_path,
                        camera_path,
                        label_path,
                        task_number]
            
            elif signal_type == 'mt':
                start_index_throat = time_start * fs_throat
                stop_index_throat = time_finish * fs_throat
                time_array = np.arange(time_start,time_finish,1/fs)
                
                data_audio_throat_segment = data_audio_throat[int(start_index_throat):int(stop_index_throat)+1]
                diference = len(time_array)-len(data_audio_throat_segment)+1
                data_audio_throat_segment = data_audio_throat[int(start_index_throat):int(stop_index_throat)+diference]

                data_x_segment = data_x_axis[int(start_index_throat):int(stop_index_throat)+diference]
                data_y_segment = data_y_axis[int(start_index_throat):int(stop_index_throat)+diference]
                data_z_segment = data_z_axis[int(start_index_throat):int(stop_index_throat)+diference]
                
                task_number = str(audio_path.split('/')[-1].split('_')[0])
                label = label.replace('\n','')
                fs = fs_throat
                # STORE INFORMATION IN ARRAY TO TRANSFORM IT INTO A DATAFRAME
                info = [index,
                        subject_id,
                        time_start,
                        time_finish,
                        time_array,
                        fs,
                        data_audio_throat_segment,
                        data_x_segment,
                        data_y_segment,
                        data_z_segment,
                        label,
                        audio_path,
                        throat_mic_path,
                        camera_path,
                        label_path,
                        task_number]
            elif signal_type == 'm':
                start_index_condenser = time_start * fs_condenser
                stop_index_condenser = time_finish * fs_condenser
                fs = fs_condenser
                time_array = np.arange(time_start,time_finish,1/fs)
                
                data_audio_condenser_segment = data_audio_condenser[int(start_index_condenser):int(stop_index_condenser)+1]
                diference = len(time_array)-len(data_audio_condenser_segment)+1
                data_audio_condenser_segment = data_audio_condenser[int(start_index_condenser):int(stop_index_condenser)+diference]


                data_x_segment = data_x_axis[int(start_index_condenser):int(stop_index_condenser)+diference]
                data_y_segment = data_y_axis[int(start_index_condenser):int(stop_index_condenser)+diference]
                data_z_segment = data_z_axis[int(start_index_condenser):int(stop_index_condenser)+diference]
                
                task_number = str(audio_path.split('/')[-1].split('_')[0])
                label = label.replace('\n','')
                
                # STORE INFORMATION IN ARRAY TO TRANSFORM IT INTO A DATAFRAME
                info = [index,
                        subject_id,
                        time_start,
                        time_finish,
                        time_array,
                        fs,
                        data_audio_condenser_segment,
                        data_x_segment,
                        data_y_segment,
                        data_z_segment,
                        label,
                        audio_path,
                        condenser_mic_path,
                        camera_path,
                        label_path,
                        task_number]
            info_array.append(info)
            index = index + 1
    print("Dataset created succesfully")
    return info_array

def create_dataset(signal_type = 'T'):
    name = f'information_dataset_{signal_type}.pickle'
    signal_type = signal_type.lower()
    if os.path.exists(os.path.join(RESULTS_PATH,name)):
        print("File already exist, no need to create a new one")
        with open(os.path.join(RESULTS_PATH,name), 'rb') as output:
            info_dataframe = pickle.load(output)
            print("File loaded succesfully")
    else:
        info_array = read_audio_info(RECORDINGS_PATH,signal_type)
        if signal_type == 't':
            info_dataframe = pd.DataFrame(info_array,columns=['Seg','ID','time_start','time_finish','time_array','fs','throat_segment','condenser_segment','label','signals_path','throat_mic_path','condenser_mic_path','camera_path','label_file_path','task_number'])
        elif signal_type == 'mt':
            info_dataframe = pd.DataFrame(info_array,columns=['Seg','ID','time_start','time_finish','time_array','fs','throat_segment','x_segment','y_segment','z_segment','label','signals_path','throat_mic_path','camera_path','label_file_path','task_number'])
        elif signal_type == 'm':
            info_dataframe = pd.DataFrame(info_array,columns=['Seg','ID','time_start','time_finish','time_array','fs','condenser_segment','x_segment','y_segment','z_segment','label','signals_path','condenser_mic_path','camera_path','label_file_path','task_number'])
        else:
            print("Pleas choose a correct signal type, either: T, MT, M") 
            return
        info_dataframe.set_index('Seg', inplace=True)
        
        with open(os.path.join(RESULTS_PATH,name), 'wb') as output:
            pickle.dump(info_dataframe,output) 
        print(f"Dataset stored succesfully as {name}")
    return info_dataframe

def read_info_dataset(dataset, audio_type='throat', hidden = True, random = True, segment = 1):
  if audio_type == 'throat':
    if random == False:
      data = dataset.loc[segment]
      fs = data['fs']
      time_start = data['time_start']
      time_finish = data['time_finish']
      subject = data['ID']
      label = data['label']
      t = data['time_array']
      t = t - np.min(t)
      signal = data['throat_signal_segment']

      fig_1= plt.figure(figsize=(15,10))
      if not hidden:
        plt.figtext(.7, .8, f"Label = {label}",fontsize = 'xx-large')
        plt.figtext(.7, .75, f"Subject = {subject}",fontsize = 'xx-large')

      plt.title("Throat Mic signal")
      plt.ylim(-1.1, 1.1)
      plt.xticks(np.arange(min(t), max(t), 0.1))

      plt.ylabel("Amplitude")
      plt.xlabel("Time (s)")
      plt.plot(t,signal)
      plt.show()
      #PLOT SIGNAL IN FREQUENCY DOMAIN
      fig, ax = plt.subplots(figsize=(15,10))
      D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
      img = librosa.display.specshow(D, y_axis='log', x_axis='time',sr=fs, ax = ax)
      print(np.array(D).shape)
      fig.colorbar(img,ax = ax, format="%+2.f dB")
      plt.show()
  else:
      data = dataset.loc[segment]
      fs = data['fs']
      time_start = data['time_start']
      time_finish = data['time_finish']
      subject = data['ID']
      label = data['label']
      t = data['time_array']
      signal = data['condenser_signal_segment']
      rms_level=0
      r = 10**(rms_level / 10.0)

      a = np.sqrt( (len(signal) * r**2) / np.sum(signal**2) )

      signal_normal = signal * a
      fig_1 = plt.figure(figsize=(15,10))
      #PLOT SIGNAL IN TIME DOMAIN
      fig_1 = plt.figure(figsize=(15,10))
      if not hidden:
        plt.figtext(.7, .8, f"Label = {label}")
        plt.figtext(.7, .75, f"Subject = {subject}")

      plt.title("Condenser Mic signal")
      plt.ylim(-1, 1)
      plt.ylabel("Amplitude")
      plt.xlabel("Time (s)")
      plt.plot(t,signal_normal)
      plt.show()

      #PLOT SIGNAL IN FREQUENCY DOMAIN
      fig, ax = plt.subplots(figsize=(15,10))
      D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
      img = librosa.display.specshow(D, y_axis='log', x_axis='time',sr=fs, ax = ax)
      fig.colorbar(img,ax = ax, format="%+2.f dB")
      plt.show()