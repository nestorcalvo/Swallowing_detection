
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pathlib import Path
import glob
import itertools
from scipy.io import wavfile
from scipy.signal import correlate
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from dtaidistance import dtw
import scipy.stats as stats

def normalize_signal(y):
  y = y[1200:-1200]
  # Calculate the maximum amplitude
  max_amplitude = np.max(np.abs(y))

  # Set desired maximum peak amplitude (e.g., 0 dBFS = 1.0)
  target_amplitude = 1.0

  # Normalize the audio to the target amplitude
  normalized_audio = y * (target_amplitude / max_amplitude)

#   normalized_audio = normalized_audio + np.min(normalized_audio)

  # Calculate the mean amplitude (DC offset)
#   mean_amplitude = np.mean(normalized_audio)

#   # Center the audio around zero by subtracting the mean amplitude
#   normalized_audio = normalized_audio - mean_amplitude
  # normalized_audio = normalized_audio[1200:-1200]
  return normalized_audio

def cross_correlate(audio1, audio2):
    correlation = correlate(audio1, audio2, mode='full')
    return correlation

def plot_correlation(correlation, title):
    plt.plot(correlation)
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.show()

def phoneme_distance(df, subjects, phoneme, task = '008_phrase_mt'):
    df_results = pd.DataFrame(columns=subjects, index=subjects)

    for subject_1 in subjects:
        for subject_2 in subjects:
            audio1 = df.loc[subject_1,task, phoneme]['segment']

            audio2 = df.loc[subject_2,task, phoneme]['segment']
            #correlation = cross_correlate(audio1, audio2)
            #lag = np.argmax(correlation) - (len(audio1) - 1)
            # similarity = cosine_similarity([audio1], [audio2])
            # distance = cosine_distances([audio1], [audio2])
            dtw_value = dtw.distance(audio1, audio2)
            df_results[subject_2][subject_1] = dtw_value

    return df_results

GENERAL_PATH = 'Recordings'
subject_folder = [f.path for f in os.scandir(GENERAL_PATH) if f.is_dir()]
tasks = ['008_phrase_mt'] #, '007_vowel_mt'

combinations = list(itertools.product(subject_folder, tasks))
# figure = plt.figure()
audio_data = []
subjects_data = []
subject_selected = []
sample_rates = []
results = {}
for i, (subject, task) in enumerate(combinations):
    
    subject_id = int(subject.split('/')[1])
    results[subject_id] = {}
    results[subject_id][task] = {}
    path_phonation_info = os.path.join(subject,task,'grouped')
    if os.path.exists(path_phonation_info) == False:
        continue

    
    task_name = task.split('_')[1]
    subject_selected.append(subject_id)
    signal_csv_path = os.path.join(path_phonation_info, f'signal_{subject_id}_{task_name}.csv')
    signal_wav_path = os.path.join(path_phonation_info, f'signal_{subject_id}_{task_name}.wav')
    magnetic_wav_path = os.path.join(path_phonation_info, f'magnetic_signal_{subject_id}_{task_name}.wav')
    
    samplerate, signal_wav = wavfile.read(signal_wav_path)
    samplerate_mag, magnetic_wav = wavfile.read(magnetic_wav_path)
    signal_wav = normalize_signal(signal_wav)
    sample_rates.append(samplerate_mag)
    
    magnetic_wav = normalize_signal(magnetic_wav)
    
    signal_csv = pd.read_csv(signal_csv_path, delimiter=';')
    
    phonation_segments = signal_csv['MAU'].values
    
    all_phonations = list(set(phonation_segments))
    for phonation in all_phonations:
        segment_information = signal_csv[['BEGIN','DURATION']][signal_csv['MAU']==phonation]
        phoneme_number = 0
        # print(segment_information['BEGIN'].iloc[phoneme_number])
        begin = segment_information['BEGIN'].iloc[phoneme_number]
        
        
        duration = segment_information['BEGIN'].iloc[phoneme_number] + segment_information['DURATION'].iloc[phoneme_number]
        
        
        magnetic_wav_short = magnetic_wav[begin:duration]
        # audio_data.append(magnetic_wav)
        # subjects_data.append(subject_id)
        results[subject_id][task][phonation] = {
            'segment': magnetic_wav_short,
            'begin': begin,
            'duration': duration,
            'phoneme_number': phoneme_number
        }
     
df = pd.DataFrame.from_dict(
    {(level1, level2, level3): value for level1, inner_dict in results.items() 
                             for level2, inner_inner_dict in inner_dict.items() 
                             for level3, value in inner_inner_dict.items()}, orient = 'index')
print(df)

all_phonemes_found = []
for indexes in df.index:
    all_phonemes_found.append(indexes[2])
all_phonemes_found = list(set(all_phonemes_found))
print(all_phonemes_found)


# for (i,(audio1,s1)), (j,(audio2,s2)) in itertools.combinations(enumerate(zip(audio_data,subjects_data)), 2):
#     correlation = cross_correlate(audio1, audio2)
#     lag = np.argmax(correlation) - (len(audio1) - 1)
#     # similarity = cosine_similarity([audio1], [audio2])
#     # distance = cosine_distances([audio1], [audio2])
#     dtw_value = dtw.distance(audio1, audio2)
#     print(dtw_value)
#     results[(i, j)] = {'correlation': correlation, 'lag': lag, 'length1': len(audio1), 'length2': len(audio2), 
#                        'subject1':s1, 'subject2':s2, 'dtw': dtw_value}

# # Plot all correlations in one plot with legends
# plt.figure(figsize=(14, 8))

# for (i, j), result in results.items():
#     correlation = result['correlation']
#     length1 = result['length1']
#     length2 = result['length2']
#     subject1 = result['subject1']
#     subject2 = result['subject2']
#     # cs = result['cs']
#     # cd = result['cd']
#     dtw_value = result['dtw']
#     lags = np.arange(-length1 + 1, length2)
#     plt.plot(lags, correlation, label=f'{subject1} vs {subject2} with dtw: {dtw_value}')

# plt.title('Cross-Correlation between Multiple Audio Signals')
# plt.xlabel('Lag')
# plt.ylabel('Correlation')
# plt.legend()
# plt.grid(True)
# plt.show()
# # Print the lags with the highest correlation for each pair
# for (i, j), result in results.items():
#     print(f"The lag with the highest correlation between {result['subject1']} and {result['subject2']} is: {result['lag']} samples")

# for phonemes_found in all_phonemes_found:
#     print(f'Distances of phoneme {phonemes_found} is being generated....')
#     task = '008_phrase_mt'
#     df_results = phoneme_distance(df, subject_selected, phonemes_found)
#     print(df_results)
#     filepath = Path(f'./Results_phonemes/results_{task}_{phonemes_found}.csv')  
#     filepath.parent.mkdir(parents=True, exist_ok=True)  
#     df_results.to_csv(filepath)
#     print(f'File being saved in {filepath}')

df_results = pd.DataFrame(columns=subject_selected, index=subject_selected)
array_segments = []
for subject_1 in subject_selected:
    
    print(subject_1)
    
    audio1 = df.loc[subject_1,task, 'O']['segment']
    #audio2 = df.loc[subject_2,task, 'O']['segment']
    array_segments.append(audio1)

lengths_array = [len(x) for x in array_segments]
minimun_length = min(lengths_array)
array_segments = [x[:minimun_length] for x in array_segments]
fvalue, pvalue = stats.f_oneway(*array_segments)
print(fvalue, pvalue)
