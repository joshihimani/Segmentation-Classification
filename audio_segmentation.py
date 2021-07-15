

import librosa
import numpy as np
import math
import librosa.display
import matplotlib.pyplot as plot

path = '/home/hp/Desktop/Speech files/should.wav'
y, sr = librosa.load(path, sr = None) # load the audio data
n = len(y)
#print(n)
fr = 100
win_len=.03
frame_length = int(np.ceil(win_len * sr))
hop_length = int(np.ceil(n/fr))
hl = hop_length/sr

frames = librosa.util.frame(y, frame_length, hop_length) 
No_of_frame=math.floor((n-frame_length)/hop_length)+1
#print(No_of_frame)

signal_energy = np.sum(np.square(frames),axis = 0)
threshold = np.mean(signal_energy) 
print("threshold calculated")
B=np.zeros(n)
for j in range(0,No_of_frame-1):
    #print(No_of_frame)
    if(j!=No_of_frame):
      if(signal_energy[j]>threshold):
         
         B[0+(j)*hop_length:0+(j+1)*hop_length] = 1
      else: 
         B[0+(j)*hop_length:0+(j+1)*hop_length] = 0

    if(j==No_of_frame):
      if(signal_energy[j]>threshold):
         B[0+(j)*hop_length:]=np.append((B,1))
      else: 
         B[0+(j)*hop_length:]=np.append((B,0))
#print(B)
    #print(B)
print("know the speech and non speech")
length=len(B)
#print(length)
newB=np.zeros(length)
j=1
k=0
if (B[1]!=B[0]):
    for  i in range(0,length-1):
        if(B[i]==B[i+1]):
            j=j+1
            k=k
        else:
            k=k+1
            j=1
        newB[k]=j
        newB[1]=1
        b=np.trim_zeros(newB)
    #print(newB)
else:
    for  i in range(0,length-1):
        if(B[i]==B[i+1]):
            j=j+1
            k=k
        else:
            k=k+1
            j=1  
        newB[k] = j 
        b=np.trim_zeros(newB)
    print(b)
print("length of Speech and non speech")
f = open('four_file', 'w')
s= len(b)
s_nsp = np.zeros((s))
zeroflag = 1
flag =0
oneflag = 0
for i in B:
    #print("iiiiiiiiiiiiiiiiiiiiiiii",i)
    if(i==0 and i!=zeroflag):
        C = i
        zeroflag=0
        oneflag =0 
        s_nsp[flag]=C
        flag = flag+1
        print("Non-Speech")
        #print(C)
        f.write("Non-Speech \n")
        
        
    if(i==1 and i!=oneflag):
        C = i
        oneflag = 1
        zeroflag = 1
        print("Speech")
        s_nsp[flag]=C
        flag = flag+1
        f.write("Speech \n")
#speech_nonspeech =np.trim_zeros(C)      
        #print(C)

print("calculate time")        
T = 1/sr
Total_time_output=np.zeros((s,2))
for i in range(0,s):  
    #print(i)
    if(i!=0):
        Total_time_output[i,0]= (np.sum(b[0:(i)])) * T
        Total_time_output[i,1]=(np.sum(b[0:(i+1)]))*T
    else:
        Total_time_output[i,0]=0*T
        Total_time_output[i,1]=(np.sum(b[0:(i+1)]))*T 
print(Total_time_output)

f.write("{} \n " .format(Total_time_output) )
f.close() 


plot.figure(figsize=(14, 5))
plot.plot(y)
plot.plot(B)
plot.show()

 

    
'''
path = '/home/hp/Desktop/Speech files/should.wav'
y, sr = librosa.load(path)
n = len(y)
fr = 100
winSize = int(np.ceil(30e-3 *sr))
#print(winSize)
hl = int(np.ceil(len(y)/fr))
hop_sec = hl/sr
#print(hop_sec)       
sigFrames = librosa.util.frame(y, frame_length = winSize, hop_length = hl) 

sigSTE = np.sum(np.square(sigFrames),axis = 0)
#print(sigSTE)
#print(sigSTE) 
mean = np.mean(sigSTE)
x = sigSTE > mean
#print(x) 
duration = 0
d = 0
T=1/sr
f = open('tryfileoutput', 'w')
B=np.zeros(n)
for j in range(0,No_of_frame-1):
    #print(No_of_frame)
    if(j!=No_of_frame):
      if(sigSTE[j]> mean):
         
         B[0+(j-1)*hl:0+(j)*hl] = 1
      else: 
         B[0+(j-1)*hl:0+(j)*hl] = 0        
    if(j==No_of_frame):
      if(sigSTE[j]> mean):
         B[0+(j-1)*hl:]=np.append((B,1))
      else: 
         B[0+(j-1)*hl:]=np.append((B,0))
#print(B)

        
speech = sigSTE[x]
npnspeech = sigSTE[~x]
#print(x)

vframes = np.array(sigFrames.flatten()[np.where(x==1)], dtype=y.dtype)
#print(vframes)
sp = np.where(vframes)
#print(sp)

B1=np.zeros(No_of_frame)
for i in range(0,No_of_frame-1):
    if(i!=No_of_frame):
        B1[i]=sum(B[(0+(i)*hl):((i+1)*hl)])/hl
    else:
        B1[i]=sum(B[(0+(i)*hl):])/winSize
#print(B1)

time=np.zeros((No_of_frame,3))

for i in range(0,No_of_frame):
    if(i!=No_of_frame):
        time[i,0]= (i)*hl*T;
        time[i,1]=(i+1)*hl*T;
    else:
        time[i,0]= (i)*hl*T;
        time[i,1]=((((i)*hl)*T)+(winSize*T))
    time[i,2]=(B1[i])
print(time)
speechIndices = np.where(speech)
nonspeechIndices = np.where(npnspeech)

f.write("{} " .format(time))
f.close()  
print("Speech Frames: ", str(speechIndices[0].shape[0]))
print("Non Speech Frames: ", str(nonspeechIndices[0].shape[0]))

vad = max(y)*x
print(vad)
#librosa.display.waveplot(y, sr=sr)

plot.subplots(1, 1, figsize=(20, 10))
plot.plot(y)
plot.plot(vad)
plot.show()
'''

