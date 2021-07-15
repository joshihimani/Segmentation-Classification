#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:49:27 2021

@author: hp
"""

from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)
x_train = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/x_train.npy')
y_train = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/data/y_train.npy')
x_test = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/x_test.npy')
y_test = np.load(f'/home/hp/.config/spyder-py3/music-speech/wavfile/tdata/y_test.npy')

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)
x_train = x_train/255
x_test = x_test/255

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train, x_val, y_train_one_hot, y_val = train_test_split(x_train, y_train_one_hot, test_size=0.2,random_state=3)

model = Sequential()
model.add(Flatten())
model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))  
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])

batch_size = 20
epochs = 20
history = model.fit(x_train, y_train_one_hot, epochs=epochs, batch_size = batch_size, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test_one_hot, batch_size=batch_size)

summary=model.summary()
print(summary)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_predictions = model.predict(x_test)
print(np.argmax(test_predictions[1]))

print("Baseline Error: %.2f%%" % (100-score[1]*100))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Classifier Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Classifier Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc = 'upper left')
plt.show()

import csv,os
header = 'filename Predictions Ground_Truth'
header = header.split()

filename = open('cnn_r.csv', 'w', newline='')
with filename:
    writer = csv.writer(filename)
    writer.writerow(header)
classes='speech music m+s'.split()
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        to_append = f'{file}'
        to_append += f' {c}'
        file = open('cnn_r.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
predicted_classes = np.argmax(np.round(test_predictions),axis=1)
#print(predicted_classes.shape)
#print(y_test.shape)
import pandas as pd
results=pd.DataFrame({"Predictions":predicted_classes, "Ground Truth":y_test})
results.to_csv("cnn__results.csv",index=False)

