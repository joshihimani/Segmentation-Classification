#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:05:55 2021

@author: hp
"""

'''
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pathlib

classes=['speech', 'music', 'm+s']
for c in classes:
   pathlib.Path(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/{c}').mkdir(parents=True, exist_ok=True)
   for filename in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train/{c}'):
        name = f'/home/hp/.config/spyder-py3/music-speech/wavfile/train/{c}/{filename}'
        y, sr = librosa.load(name)
        S=librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        #outFileName = output + filename.split('.')[0]+'.png'
        plt.savefig(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/{c}/{filename[:-3].replace(".", "")}.png')
        plt.close()
print("Train completed")

for c in classes:
   pathlib.Path(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/{c}').mkdir(parents=True, exist_ok=True)
   for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        y, sr = librosa.load(file)
        S=librosa.feature.melspectrogram(y=y, sr=sr)
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        plt.savefig(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/{c}/{files[:-3].replace(".", "")}.png')
        plt.close()
print("Test completed")


'''
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pathlib
classes=['speech', 'music', 'm+s']
for clas in classes:
    pathlib.Path(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/{clas}').mkdir(parents=True, exist_ok=True)
    files = os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train/{clas}')
    for file in files:
        filePath = f'/home/hp/.config/spyder-py3/music-speech/wavfile/train/{clas}/{file}'
        y, sr = librosa.load(filePath)
        S=librosa.feature.melspectrogram(y=y, sr=sr)
        #plt.figure(figsize=(4, 2))
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        #librosa.display.specshow(librosa.power_to_db(S,ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        #plt.colorbar(format='%+2.0f dB')
        plt.savefig(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/{clas}/{file[:-3].replace(".", "")}.png')
        plt.close()

for clas in classes:
    pathlib.Path(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/{clas}').mkdir(parents=True, exist_ok=True)
    files = os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{clas}')
    for file in files:
        filePath = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{clas}/{file}'
        y, sr = librosa.load(filePath)
        S=librosa.feature.melspectrogram(y=y, sr=sr)
        #plt.figure(figsize=(4, 2))
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        #librosa.display.specshow(librosa.power_to_db(S,ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        #plt.colorbar(format='%+2.0f dB')
        plt.savefig(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/{clas}/{file[:-3].replace(".", "")}.png')
        plt.close()

print("completed")