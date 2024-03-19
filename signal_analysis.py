import random
import re
import numpy as np
import pickle
import librosa
import matplotlib.pyplot as plt
from utils import *
from features import *


class SignalAnalysis():

    def __init__(self, dataset,signal_type):
        print("Class Inizialized")

        self.dataset = dataset
        self.hop_length = 256
        self.frame_length = 2048
        self.MELS = 128
        self.signal_type = signal_type.lower()

    def visualize_sample(self, sample_type, anonymous=True, segment=None, *args, **kwargs):
        """Function to display a sample of the database in the time and frequency domain 

        Args:
            sample_type (str): Name of the type of signal we want to analyze.
            anonymous (bool): Option to show the ID of the subject in the visualization.
            segment (int, optional): Number of the segment we want to display, None if we want a random segment.

        Raises:
            ValueError: If sample_type is not a valid value.
            ValueError: If segment select is not in the dataframe.
        """
        dataset = self.dataset

        if sample_type not in ["throat", "condenser", "magnetic"]:
            raise ValueError(
                "sample_type not available, options are: 'throat', 'condenser' or 'magnetic'")

        if segment != None and segment not in list(dataset.index):
            raise ValueError("segment selected not in dataframe")

        if not segment:
            segment_list = list(dataset.index)
            segment = random.choice(segment_list)

        
        
        data = dataset.loc[segment]

        t = data['time_array']
        t_normalized = t - np.min(t)
        signal = data.filter(regex=rf"{sample_type}_segment$").iloc[0]
        fs = data.filter(regex=rf"^fs").iloc[0]

        fig_1 = plt.figure(figsize=kwargs.get('figsize', (15, 10)))
        label = data['label']
        plt.figtext(.7, .8, f"Label = {label}", fontsize='xx-large')
        if not anonymous:
            subject = data['ID']
            plt.figtext(.7, .75, f"Subject = {subject}", fontsize='xx-large')

        name = sample_type.capitalize()
        plt.title(f"{name} Mic signal")
        plt.ylim(-1.1, 1.1)
        plt.xticks(np.arange(min(t), max(t), 0.1))
        plt.ylabel("Amplitude")
        plt.xlabel("Time (s)")
        plt.plot(t, signal)

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (15, 10)))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        img = librosa.display.specshow(
            D, y_axis='log', x_axis='time', sr=fs, ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.show()

    def store_handcrafeted_features(self, features_path_name):
        """

        Args:
            features_path_name (str): Contains string with name and extension to save features. Recomended in .pickle
        """
        print("Creating handcrafted features...")
# =============================================================================
#         if os.path.exists(features_path_name):
#             print("Features already exist, no need to create a new one")
#             return
# =============================================================================

        features = handcrafted_feature_creation(
            self.dataset, self.hop_length, self.frame_length,self.signal_type)
        print("Storing handcrafted features")
        with open(features_path_name, 'wb') as output:
            pickle.dump(features, output)
        print("Handcrafted features stored correctly")

    def store_spectograms_features(self, features_path_name):
        print("Creating spectograms features...")
        features = spectograms_feature_creation(
            self.dataset, self.hop_length, self.frame_length, self.MELS, self.signal_type)
        print("Storing spectograms features")
        with open(features_path_name, 'wb') as output:
            pickle.dump(features, output)
        print("Spectograms features stored correctly")
