#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:13:57 2024

@author: nestor
"""

import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pickle
import matplotlib.animation as animation
from scipy.signal import butter, lfilter, freqz, filtfilt,savgol_filter
from scipy.signal import hilbert


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
# Filter requirements.
order = 6
fs = 48000       # sample rate, Hz
cutoff = 40  # desired cutoff frequency of the filter, Hz
information_path = '/home/nestor/Documentos/Swallowing Detection/Results/information_dataset_M.pickle'

with open(information_path, 'rb') as output:
    info_dataframe = pickle.load(output)
    print("File loaded succesfully")



x_axis = info_dataframe.iloc[22]['x_segment']

analytic_signal = hilbert(x_axis)
 
amplitude_envelope = np.abs(analytic_signal)
y = butter_lowpass_filter(amplitude_envelope, cutoff, fs, order)
plt.plot(y)
plt.show()
y_smooth = savgol_filter(x_axis, window_length=1021, polyorder=6, mode="nearest")


#amplitude_envelope = amplitude_envelope
x_axis_t = y[10:]
x_axis_t1 = y[:-10]

R2 = np.correlate(x_axis_t, x_axis_t1, "same")


fig, ax = plt.subplots()
line2 = ax.plot(x_axis_t[0], x_axis_t1[0])[0]
ax.set(xlim=[-1, 1], ylim=[-1, 1])
def update(frame):
    
    line2.set_xdata(x_axis_t[:frame])
    line2.set_ydata(x_axis_t1[:frame])
    return line2
ani = animation.FuncAnimation(fig=fig, func=update, frames=50000, interval=10)
plt.show()