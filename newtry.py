
import librosa
import numpy as np
import math
import matplotlib.pyplot as plot

'''
path = '/home/hp/Desktop/Speech files/should.wav'
y, sr = librosa.load(path, sr = None) # load the audio data
n = len(y)
np.set_printoptions(threshold=np.inf)
#frame_rate = int(input("Enter frame rate: "))
#window_sec = float(input("Enter time: "))
#frame_length = int(np.ceil(window_sec * sr))
frame_rate = 100
frame_length = int(np.ceil(0.03 * sr))
hop_length = int(np.ceil(n/frame_rate))

hop_sec = hop_length/sr
'''
A = 15
n = np.arange(A)
print(n)
hop_length=3;
frame_length=6
sampling_rate=10
sampling_period=0.01

frames = librosa.util.frame(n, frame_length, hop_length) 
print(frames)
No_of_frame=math.floor((len(n)-frame_length)/hop_length)+1
print(No_of_frame)
#print(frames)
signal_energy = np.sum(np.square(frames),axis = 0)
print(signal_energy.shape)
threshold = np.mean(signal_energy) 
print(threshold)


B=np.zeros(len(n))

for j in range(1,No_of_frame):
    #print(No_of_frame)
    if(j!=No_of_frame):
      if(signal_energy[j]>threshold):
               
         B[0+(j-1)*hop_length:0+(j)*hop_length] = 1
      else: 
       
         B[0+(j-1)*hop_length:0+(j)*hop_length] = 0

    if(j==No_of_frame):
      if(signal_energy[j]>threshold):
         B[0+(j-1)*hop_length:]=np.append((B,1))
      else: 
         B[0+(j-1)*hop_length:]=np.append((B,0))
print(B)
length=len(B)
#print(length)
newB=np.zeros(len(n))

T = 0.01
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
    #print(b)
size = len(newB)
#print("length", size)  
s = len(b) 

zeroflag = 1
oneflag = 0
flag = 0
flag1 = 0

s_nsp = np.zeros((s))
for i in B:
   
    if(i==0 and i!=zeroflag):
        C = i
        zeroflag=0
        oneflag =0 
        s_nsp[flag]=C
        flag = flag+1
        #print("Non-Speech")
        #print(C)       
    if(i==1 and i!=oneflag):
        C = i
        oneflag = 1
        zeroflag = 1
        s_nsp[flag]=C
        flag = flag+1       
        #print("Speech")    
        #print(C) 
     
Total_time_output=np.zeros((s,3))

for i in range(0,s):  
    #print(i)
    if(i!=0):
        Total_time_output[i,0]= (np.sum(b[0:(i)])) * T
        Total_time_output[i,1]=(np.sum(b[0:(i+1)]))*T
    else:
        Total_time_output[i,0]=0*T
        Total_time_output[i,1]=(np.sum(b[0:(i+1)]))*T
  
print(Total_time_output)

