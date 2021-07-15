import librosa
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot
import librosa.display

np.set_printoptions(threshold=np.inf)

def mfcc(y,sr,hl):
    mfcc = librosa.feature.mfcc(y,sr=sr,hop_length=hl)
    print(mfcc.shape)
    new = np.transpose(mfcc)
    print(new.shape) 
    kmeans = KMeans(n_clusters=2,random_state=0).fit(new)
    print(kmeans.labels_)
    No_of_frame=100
    s=np.zeros(No_of_frame)
    s=kmeans.labels_
    return s,No_of_frame

def mfcc_segmentation(n, s, sr, hl,No_of_frame,win_len):
    samples=np.zeros(n)
    for j in range(0,No_of_frame):
     if(j!=No_of_frame):
      if(s[j]==1):
         samples[0+(j)*hl:0+(j+1)*hl] = 1
      else: 
         samples[0+(j)*hl:0+(j+1)*hl] = 0        
     if(j==No_of_frame):
      if(s[j]==0):
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
    return time,samples

def framing(frame_length, hl,y):
    sigFrames = librosa.util.frame(y, frame_length = frame_length, hop_length = hl) 
    return sigFrames

def energy(sigFrames):
    signal_energy = np.sum(np.square(sigFrames),axis = 0)
    print(signal_energy.shape)
    frame=97
    No_frame = frame-1
    sig=signal_energy.reshape(-1,1)
    print(sig.shape)
    kmeans1 = KMeans(n_clusters=2, random_state=0).fit(sig)
    print(kmeans1.labels_)
    s_energy=np.zeros(No_frame)
    s_energy=kmeans1.labels_
    return s_energy, No_frame

def energy_segmentation(hl, n, win_len, s_energy, No_frame, sr):
    e_samples=np.zeros(n)
    for j in range(0,No_frame):
    #print(No_of_frame)
     if(j!=No_frame):
      if(s_energy[j]==1):
        
         e_samples[0+(j)*hl:0+(j+1)*hl] = 1
      else: 
         e_samples[0+(j)*hl:0+(j+1)*hl] = 0        
     if(j==No_frame):
      if(s_energy[j]==1):
         e_samples[0+(j)*hl:]=1
      else: 
         e_samples[0+(j)*hl:]=0
#print(samples)

    T= 1/sr
    e_time=np.zeros((No_frame,3))
    for i in range(0,No_frame):
      if(i!=No_frame):
        e_time[i,0]= (i)*hl*T;
        e_time[i,1]=(i+1)*hl*T;
      else:
        e_time[i,0]= (i)*hl*T;
        e_time[i,1]=((((i)*hl)*T)+(win_len*T))
      e_time[i,2]=(s_energy[i])
    return e_time,e_samples

def truth_values(n):
   B=np.zeros(n)
   for i in range(0,n):
    if (i<=1808):
        B[i]=0
    elif(i<5534 and i>=1808):
        B[i]=1
    elif(i<6945 and i>=5534):
        B[i]=0
    elif(i<12479 and i>=6945):
        B[i]=1
    elif(i<20100 and i>=12479):
        B[i]=0
    elif(i<33955 and i>=20100):
        B[i]=1
    elif(i<38806 and i>=33955):
        B[i]=0
   return B

if __name__ == "__main__":
    path = '/home/hp/Desktop/Speech files/f1.wav'
    y, sr = librosa.load(path) # load the audio data
    n = len(y)
    #print(sr)
    fr = 100
    win_len=.03
    frame_length = int(np.ceil(win_len * sr))
    hl = int(np.ceil(n/fr))
    s, No_of_frame=mfcc(y,sr,hl)
    time,samples= mfcc_segmentation(n, s, sr, hl,No_of_frame,win_len)
    B=truth_values(n)
    sigFrames=framing(frame_length, hl,y)
    s_energy,No_frame=energy(sigFrames)
    e_time,e_samples=energy_segmentation(hl, n, win_len, s_energy, No_frame, sr)
    plot.subplots(figsize=(15, 10))
    plot.plot(y) 
    plot.plot(samples,color="r",label="MFCC Segmentation")
    plot.plot(e_samples,color="m",label="Energy Segmentation")
    plot.plot(B,color="g",label="True Segmentation")
    plot.legend(loc="best")

    plot.show()
    from sklearn.metrics import f1_score
    print("The F1-score is: ", f1_score(samples, B, average='weighted'))
    print("The F1-score of energy is: ", f1_score(e_samples, B, average='weighted'))    
    