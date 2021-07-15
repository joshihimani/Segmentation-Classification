#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:40:10 2021

@author: hp
"""
import librosa
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot
path = '/home/hp/Desktop/Speech files/this.wav'
y, sr = librosa.load(path) # load the audio data
print(sr)
n = len(y)
#print(n)
fr = 100
win_len=.03
frame_length = int(np.ceil(win_len * sr))
hl = int(np.ceil(n/fr))


#frame=math.floor((n-win_len)/hl)+1  
frame=97
No_of_frame = frame-1
#print(No_of_frame)
frames = librosa.util.frame(y, frame_length, hl) 
signal_energy = np.sum(np.square(frames),axis = 0)

sig=signal_energy.reshape(-1,1)
print(sig.shape)
kmeans1 = KMeans(n_clusters=2, random_state=0).fit(sig)
print(kmeans1.labels_)

s=np.zeros(No_of_frame)
s=kmeans1.labels_
#print(s)
np.set_printoptions(threshold=np.inf)
samples=np.zeros(n)
for j in range(0,No_of_frame):
    #print(No_of_frame)
     if(j!=No_of_frame):
      if(s[j]==1):
        
         samples[0+(j)*hl:0+(j+1)*hl] = 1
      else: 
         samples[0+(j)*hl:0+(j+1)*hl] = 0        
     if(j==No_of_frame):
      if(s[j]==1):
         samples[0+(j)*hl:]=1
      else: 
         samples[0+(j)*hl:]=0
#print(samples)

T= 1/sr
time=np.zeros((No_of_frame,3))
for i in range(0,No_of_frame):
      if(i!=No_of_frame):
        time[i,0]= (i)*hl*T;
        time[i,1]=(i+1)*hl*T;
      else:
        time[i,0]= (i)*hl*T;
        time[i,1]=((((i)*hl)*T)+(win_len*T))
      time[i,2]=(s[i])
#print(time)

plot.plot(y,color="g") 


plot.plot(samples,color="r")
plot.show()

B=np.zeros(n)
for i in range(0,n):
    if (i<=2513):
        B[i]=0
    elif(i<15169 and i>=2513):
        B[i]=1
    elif(i<16447 and i>=15169):
        B[i]=0
    elif(i<21915 and i>=16447):
        B[i]=1
    elif(i<23215 and i>=21915):
        B[i]=0
    elif(i<26279 and i>=23215):
        B[i]=1
    elif(i<30336 and i>=26279):
        B[i]=0
    elif(i<41802 and i>=30336):
        B[i]=1
    elif(i<42419 and i>=41802):
        B[i]=0

plot.plot(y,color='g') 
plot.plot(samples,color="b")
plot.plot(B,color='r')
plot.show()
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
print("The recall score is: ", recall_score(samples, B, average='weighted'))
print("The precision score is: ", precision_score(samples, B, average='weighted'))
print("The F1-score is: ", f1_score(samples, B, average='weighted'))

