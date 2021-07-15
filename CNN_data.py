#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:42:00 2021

@author: hp
"""
import os
import numpy as np
import keras
from tensorflow.keras.applications.vgg16 import VGG16
import cv2

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))




X_train = []
Y_train = []

classes=['speech', 'music', 'm+s']
input_path = f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/' 
for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = np.expand_dims(img, axis=0)
        X_train.append(np.squeeze(img))
        Y_train.append(classes.index(clas))
        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(X_train.shape)

#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]))
np.save(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/X_train.npy', X_train)
np.save(f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/Y_train.npy', Y_train)


input_path = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/' 

X_test = []
Y_test = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = np.expand_dims(img, axis=0)
        X_test.append(np.squeeze(img))
        Y_test.append(classes.index(clas))


X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
print(X_test.shape)

#x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))
np.save(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/X_test.npy', X_test)
np.save(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/Y_test.npy', Y_test) 








