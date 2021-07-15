#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:13:23 2021

@author: hp
"""


frame_rate = input("Enter the frame rate: ")
frame_rate = int(frame_rate)

#print("Total number of frames in one second :", frame_rate)

fs = input("Enter the sampling rate: ")
fs = int(fs)

duration = input("Enter the duration: ")
duration = int(duration)

samples = fs * duration
print("Samples :", samples)

import numpy as np
n = np.arange(samples)
size = len(n)
np.set_printoptions(threshold=np.inf)
print(f'n = {n}')

H = samples / frame_rate 
H=int(H)
hop_length = H
#hop_length = H * duration
print("The hop length for the samples is: ", hop_length)

window_sec = input("Enter the duration for window: ")
window_size = float(window_sec) * fs
frame_length = int(window_size)
print(frame_length)

import librosa
frames = librosa.util.frame(n, frame_length = frame_length, hop_length = hop_length, axis=0)
print(frames)
print("Shape of the frames",frames.shape)

f = open("Samples_file.txt","w+")
f.write(str(frames))
f.close()



