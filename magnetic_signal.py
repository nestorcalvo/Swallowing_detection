from utils import *
from constant import *
from scipy.io.wavfile import write
import soundfile
import pathlib

def magnetic_signal_cleannig(signal, fs):
    fc = 40
    filter_order = 6
    signal = np.abs(hilbert(signal))
    signal = butter_lowpass_filter(signal, fc, fs, filter_order)
    signal = signal[1200:-1200]
    return signal
 
signal_type = 'MT'
full_classes = True
LOSO = True

class_ammount = 'five' if full_classes else 'three'
cross_validation_type = 'LOSO' if LOSO else 'SGKFold'

signal_type = signal_type.lower()
if signal_type == 't':
    pattern = '_t/'
    other_signal_pattern = '_m/'
elif signal_type == 'm':
    pattern = '_m/'
    other_signal_pattern = '_t/'
else:
    print('Early fusion modality')
    pattern = '_mt/'
reg_path = f'/**/*_{signal_type}/*'
files = glob.glob(RECORDINGS_PATH +reg_path, recursive = True)
# print(files)

info_array = []
index = 1
# Find the folder name of a task for a type of signal
regex_instance = f'[0-9]*_(phrase|vowel)_{signal_type}/'
p = re.compile(regex_instance)
print("Creating dataset with segments...")
count = 0
sub_temp = 0
task_temp = 0
for f in files:
    result = p.search(f)
    
    if result:
        count += 1
        match = result.group(0)
        f = os.path.normpath(f)
        subject_id = int(f.split(os.sep)[-3])
        if subject_id != 2:
            print(subject_id)
            print(f)
            continue
        audio_path = os.path.dirname(os.path.abspath(f))
        task = audio_path.split('_')[1]
        if subject_id == sub_temp and task == task_temp:
            continue
        else:
            sub_temp = subject_id
            task_temp = task
            print(subject_id)
            print(task)
        if signal_type == 't':
            throat_mic_path = os.path.join(audio_path, 'throat_mic.wav')
            condenser_mic_path = os.path.join(audio_path, 'condenser_mic.wav')
            
            fs, _ = read(throat_mic_path)
            data_audio_throat, fs_throat  = librosa.load(throat_mic_path, sr = fs)
            data_audio_condenser, fs_condenser = librosa.load(condenser_mic_path, sr = fs)
        elif signal_type == 'mt':
            throat_mic_path = os.path.join(audio_path, 'throat_mic.wav')
            fs, _ = read(throat_mic_path)
            fc = 40
            filter_order = 6
            data_audio_throat, fs_throat  = librosa.load(throat_mic_path, sr = fs)
            
            x_axis_path = os.path.join(audio_path, 'x_axis.wav')
            y_axis_path = os.path.join(audio_path, 'y_axis.wav')
            z_axis_path = os.path.join(audio_path, 'z_axis.wav')
            
            fs, _ = read(x_axis_path)
            data_x_axis, fs_x_axis  = librosa.load(x_axis_path, sr = fs)
            x = magnetic_signal_cleannig(data_x_axis, fs)
            data_y_axis, fs_y_axis  = librosa.load(y_axis_path, sr = fs)
            y = magnetic_signal_cleannig(data_y_axis, fs)
            data_z_axis, fs_z_axis  = librosa.load(z_axis_path, sr = fs)
            z = magnetic_signal_cleannig(data_z_axis, fs)
            result = x**2 + y**2 + z**2
            folder_signals = os.path.join(audio_path, 'grouped')
            path = pathlib.Path(folder_signals)
            path.mkdir(parents=True, exist_ok=True)

            if task == 'vowel':
                soundfile.write(os.path.join(folder_signals, f'magnetic_signal_{subject_id}_vowel.wav'),result, fs)
                soundfile.write(os.path.join(folder_signals, f'signal_{subject_id}_vowel.wav'),data_audio_throat, fs, subtype='PCM_16')
                f = open(os.path.join(folder_signals, f'signal_{subject_id}_vowel.txt'), 'w') 
                f.write('aaaaaaaaaaaa')
                f.close()
            else:
                soundfile.write(os.path.join(folder_signals, f'magnetic_signal_{subject_id}_phrase.wav'), result,fs)
                soundfile.write(os.path.join(folder_signals, f'signal_{subject_id}_phrase.wav'),data_audio_throat ,fs)
                f = open(os.path.join(folder_signals, f'signal_{subject_id}_phrase.txt'), 'w')
                f.write('Guten Morgen, wie geht es Ihnen?\nGuten Morgen, wie geht es Ihnen?')
                f.close()
        else:
            continue    

    
# print(data_x_axis)
# print(result)
# print(x)
# plt.plot(x)
# plt.plot(result)
# plt.show()