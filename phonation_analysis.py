import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import glob
import itertools
def normalize_signal(y):
  # Calculate the maximum amplitude
  max_amplitude = np.max(np.abs(y))

  # Set desired maximum peak amplitude (e.g., 0 dBFS = 1.0)
  target_amplitude = 1.0

  # Normalize the audio to the target amplitude
  normalized_audio = y * (target_amplitude / max_amplitude)

  # Calculate the mean amplitude (DC offset)
  mean_amplitude = np.mean(normalized_audio)

  # Center the audio around zero by subtracting the mean amplitude
  normalized_audio = normalized_audio - mean_amplitude
  return normalized_audio


from scipy.io import wavfile

GENERAL_PATH = 'Recordings'
subject_folder = [f.path for f in os.scandir(GENERAL_PATH) if f.is_dir()]
tasks = ['008_phrase_mt'] #, '007_vowel_mt'

combinations = list(itertools.product(subject_folder, tasks))
figure = plt.figure()
for i, (subject, task) in enumerate(combinations):
    subject_id = int(subject.split('/')[1])
    task_name = task.split('_')[1]
    path_phonation_info = os.path.join(subject,task,'grouped')
    if os.path.exists(path_phonation_info) == False:
        continue
    signal_csv_path = os.path.join(path_phonation_info, f'signal_{subject_id}_{task_name}.csv')
    signal_wav_path = os.path.join(path_phonation_info, f'signal_{subject_id}_{task_name}.wav')
    magnetic_wav_path = os.path.join(path_phonation_info, f'magnetic_signal_{subject_id}_{task_name}.wav')
    print(path_phonation_info)
    print(magnetic_wav_path)
    samplerate, signal_wav = wavfile.read(signal_wav_path)
    samplerate_mag, magnetic_wav = wavfile.read(magnetic_wav_path)
    signal_wav = normalize_signal(signal_wav)
    magnetic_wav = normalize_signal(magnetic_wav)
    
    signal_csv = pd.read_csv(signal_csv_path, delimiter=';')
    phonation_segments = signal_csv['MAU'].values
    print(signal_csv)
    print(phonation_segments)
    segment_information = signal_csv[['BEGIN','DURATION']][signal_csv['MAU']=='g']
    print(segment_information)
    phoneme_number = 3
    print(segment_information['BEGIN'].iloc[phoneme_number])
    begin = segment_information['BEGIN'].iloc[phoneme_number]
    
    duration = segment_information['BEGIN'].iloc[phoneme_number] + segment_information['DURATION'].iloc[phoneme_number]
    #plt.plot(signal_wav[begin:begin + duration])
    plt.plot(magnetic_wav[begin:duration], label = subject_id)
    plt.legend()
    #plt.title('First full phrase')
    # if i == 8:
    #     break

plt.show()
