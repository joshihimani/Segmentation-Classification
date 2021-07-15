#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:24:13 2021

@author: hp
"""

import librosa
import numpy as np
import matplotlib.pyplot as plot

path = '/home/hp/Desktop/Speech files/reach_here.wav'
y, sr = librosa.load(path)
print(sr)
n = len(y)
print(n)
B=np.zeros(n)
'''
#f1
for i in range(0,n):
    if (i<=1808):
        B[i]=0
    elif(i<5534 and i>=1808):
        B[i]=1
    elif(i<6945 and i>=5534):
        B[i]=0
    elif(i<12479 and i>=6945):
        B[i]=1
    elif(i<26613 and i>=12479):
        B[i]=0
    elif(i<33955 and i>=26613):
        B[i]=1
    elif(i<38806 and i>=33955):
        B[i]=0
'''
#print(B)
#should.wav
'''
for i in range(0,n):
    if (i<=3969):
        B[i]=0
    elif(i<6218 and i>=3969):
        B[i]=1
    elif(i<8048 and i>=6218):
        B[i]=0
    elif(i<10407 and i>=8085):
        B[i]=1
    elif(i<13273 and i>=10407):
        B[i]=0
    elif(i<17638 and i>=13273):
        B[i]=1
    elif(i<22224 and i>=17638):
        B[i]=0
'''
for i in range(0,n):
    if (i<=7430):
        B[i]=1
    elif(i<18807 and i>=7430):
        B[i]=0
    elif(i<28420 and i>=18807):
        B[i]=1
    elif(i<31993 and i>=28421):
        B[i]=0

plot.subplots(1, figsize=(10, 10))
plot.plot(y) 
    #plot.show()
plot.plot(B)
plot.show()

'''
#two
for i in range(0,n):
    if (i<=2954):
        B[i]=0
    '''
'''
    elif(i<8854 and i>=2954):
        B[i]=1
    elif(i<10022 and i>=8854):
        B[i]=0
'''
'''
    elif(i<15688 and i>=2294):
        B[i]=1
    elif(i<42258 and i>=15688):
        B[i]=0
    elif(i<53966 and i>=42258):
        B[i]=1
    elif(i<59663 and i>=53966):
        B[i]=0
        
#words
for i in range(0,n):
    if (i<=11157):
        B[i]=0
    elif(i<29524 and i>=11157):
        B[i]=1
    elif(i<52764 and i>=29524):
        B[i]=0
    elif(i<73468 and i>=52764):
        B[i]=1
    elif(i<92172 and i>=73468):
        B[i]=0
    elif(i<111459 and i>=92717):
        B[i]=1
    elif(i<124402 and i>=111459):
        B[i]=0

#listen
for i in range(0,n):
    if (i<=4652):
        B[i]=0
    elif(i<22556 and i>=4652:
        B[i]=1
    elif(i<44142 and i>=22556):
        B[i]=0
    elif(i<63391 and i>=44142):
        B[i]=1
    elif(i<81626 and i>=63391):
        B[i]=0
   
#count
for i in range(0,n):
    if (i<=)5777):
        B[i]=0
    elif(i<17772 and i>=5777):
        B[i]=1
    elif(i<42534 and i>=17772:
        B[i]=0
    elif(i<52897 and i>=42534):
        B[i]=1
    elif(i<72675 and i>=52897):
        B[i]=0
    elif(i<84618 and i>=72645):
        B[i]=1
    elif(i<92974 and i>=111459):
        B[i]=0

#test
for i in range(0,n):
    if (i<=859):
        B[i]=0
    elif(i<13273 and i>=859):
        B[i]=1
    elif(i<15566 and i>=13273):
        B[i]=0
    elif(i<25334 and i>=15566):
        B[i]=1
    elif(i<26833 and i>=24334):
        B[i]=0
    elif(i<29148 and i>=26833):
        B[i]=1
    elif(i<31000 and i>=29148):
        B[i]=0  
        
#one
for i in range(0,n):
    if (i<=2712):
        B[i]=0
    elif(i<8665 and i>=2712):
        B[i]=1
    elif(i<10076 and i>=8665):
        B[i]=0
    elif(i<15522 and i>=10076):
        B[i]=1
    elif(i<18520 and i>=15522):
        B[i]=0
    
 # this
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
'''
        

