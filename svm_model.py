#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:29:53 2021

@author: hp
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


data = pd.read_csv('dataset.csv')
data.head()

class_list = data.iloc[:, -1]
encoder = LabelEncoder()
y_train = encoder.fit_transform(class_list)

x_train = data.drop('label', axis=1)
print(x_train.shape)
print(y_train.shape)


data = pd.read_csv('dataset1.csv')
data.head()

class_list = data.iloc[:, -1]
encoder = LabelEncoder()
y_test = encoder.fit_transform(class_list)

x_test = data.drop('label', axis=1)

print(x_test.shape)
print(y_test.shape)
'''
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

svclassifier = SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)

test_predictions = svclassifier.predict(x_test)
print(test_predictions[1])

predicted_classes = test_predictions
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test,test_predictions))



import csv 
import os

header = 'filename Predictions Ground_Truth'
header = header.split()

filename = open('svm_r.csv', 'w', newline='')
with filename:
    writer = csv.writer(filename)
    writer.writerow(header)
classes='speech music m+s'.split()
for c in classes:    
  for files in os.listdir(f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}'):
        file = f'/home/hp/.config/spyder-py3/music-speech/wavfile/test/{c}/{files}'
        to_append = f'{file}'
        to_append += f' {c}'
        file = open('svm_r.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


#print(predicted_classes.shape)
#print(y_test.shape)
import pandas as pd
results=pd.DataFrame({"Predictions":predicted_classes, "Ground Truth":y_test})
results.to_csv("pred_results.csv",index=False)
'''

from keras import layers
from keras import Input
from keras.layers import Dense, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train, x_val, y_train_one_hot, y_val = train_test_split(x_train, y_train_one_hot, test_size=0.2,random_state=3)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])

batch_size = 15
epochs = 20
history = model.fit(x_train, y_train_one_hot, epochs=epochs, batch_size = batch_size, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test_one_hot, batch_size=15)

summary=model.summary()
print(summary)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_predictions = model.predict(x_test)
print(np.argmax(test_predictions[1]))

print("Baseline Error: %.2f%%" % (100-score[1]*100))

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
