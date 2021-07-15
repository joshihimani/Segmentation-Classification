#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:12:48 2021

@author: hp
"""


fs = input("Enter the sampling rate: ")
fs = int(fs)

duration = input("Enter the duration: ")
duration = int(duration)

samples = fs * duration
print("Samples per sec:", samples)


import numpy as np
n = np.arange(samples)
print(f'n = {n}')
'''
n = np.empty((0,samples))
for i in range(samples):
    n = np.append(n,i)
print(f'n = {n}')
'''
frame_length = input("Enter the frame length: ")
frame_length = int(frame_length)
hop_length= input("Enter the hop length: ")
hop_length = int(hop_length)

import librosa
frames = librosa.util.frame(n, frame_length, hop_length, axis=0)
print(frames)



f = open("samples_to_frames.txt","w+")
f.write(str(frames))
f.close()



'''
#np.set_printoptions(threshold=np.inf)
'''