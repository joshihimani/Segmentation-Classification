#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:27:54 2021

@author: hp
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import VGG16

train_dir=f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/' 
test_dir=f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/' 

BATCH_SIZE = 20                   
IMG_SIZE = (150,150)


train_dataset = image_dataset_from_directory(train_dir,
                                             label_mode='categorical',
                                             shuffle=False,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

test_dataset = image_dataset_from_directory(test_dir,
                                            label_mode='categorical',
                                             shuffle=False,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

train_batches = tf.data.experimental.cardinality(test_dataset)       # 1000
val_dataset = train_dataset.take(train_batches // 2)         #   200
train_dataset = train_dataset.skip(train_batches // 2) 

#class_names = train_dataset.class_names

print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
#print("Class Names:", class_names)

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

IMG_SHAPE = IMG_SIZE + (3,)  
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1,input_shape = IMG_SHAPE)


base_model = VGG16(weights='imagenet',
                  include_top=False,              
                  input_shape=IMG_SHAPE)
base_model.trainable = False


model = tf.keras.Sequential()
model.add(rescale)
model.add(base_model)
model.add(tf.keras.layers.Conv2D(64, (3, 3),  strides=(1, 1), padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3),  strides=(1, 1), padding = 'same', activation = 'relu'))    #   161 161 32
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding = 'valid'))    
model.add(tf.keras.layers.GlobalAveragePooling2D())    
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(3, activation='softmax')) 

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])


history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=val_dataset)

scores = model.evaluate(test_dataset)

summary=model.summary()
print(summary)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

print("Baseline Error: %.2f%%" % (100-scores[1]*100))

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

pred = model.predict(test_dataset)
#print(pred)
predicted_class_indices=np.argmax(pred,axis=1)
import pandas as pd
results=pd.DataFrame({
                      "Predictions":predicted_class_indices})
results.to_csv("p_r.csv",index=False)
'''
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

train_dir=f'/home/hp/.config/spyder-py3/music-speech/wavfile/train_data/' 
test_dir=f'/home/hp/.config/spyder-py3/music-speech/wavfile/test_data/' 
train_datagen = ImageDataGenerator(
        rescale=1./255, # rescale all pixel values from 0-255, so aftre this step all our pixel values are in range (0,1)
        shear_range=0.2, #to apply some random tranfromations
        zoom_range=0.2, #to apply zoom
        horizontal_flip=True)
test_datagen = ImageDataGenerator(
        rescale=1./255, # rescale all pixel values from 0-255, so aftre this step all our pixel values are in range (0,1)
        shear_range=0.2, #to apply some random tranfromations
        zoom_range=0.2, #to apply zoom
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        shuffle = False)
test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical',
        shuffle = False )

IMG_SIZE=(150,150)
IMG_SHAPE = IMG_SIZE + (3,)  
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1,input_shape = IMG_SHAPE)


base_model = VGG16(weights='imagenet',
                  include_top=False,              
                  input_shape=IMG_SHAPE)
base_model.trainable = False

model = tf.keras.Sequential()
model.add(rescale)
model.add(base_model)
model.add(tf.keras.layers.Conv2D(64, (3, 3),  strides=(1, 1), padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3),  strides=(1, 1), padding = 'same', activation = 'relu'))    #   161 161 32
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding = 'valid'))    
model.add(tf.keras.layers.GlobalAveragePooling2D())    
model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(3, activation='softmax')) 


epochs = 3
batch_size = 20

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit_generator(
        training_set,
        steps_per_epoch=5,
        epochs=3)

scores=model.evaluate_generator(generator=test_set)
pred = model.predict_generator(test_set)

scores = model.evaluate(test_set)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

summary=model.summary()
pred = model.predict(test_set)


predicted_class_indices=np.argmax(pred,axis=1)

labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions = predictions[:61]
filenames=test_set.filenames

predicted_class_indices=np.argmax(pred,axis=1)
import pandas as pd
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("p_results.csv",index=False)
'''