import librosa
import numpy as np
import matplotlib.pyplot as plot
np.set_printoptions(threshold=np.inf)

path = '/home/hp/Desktop/Speech files/englishwords.wav'
y, sr = librosa.load(path)
n = len(y)
fr = 100
winSize = int(np.ceil(30e-3 *sr))
#print(winSize)
hl = int(np.ceil(len(y)/fr))
hop_sec = hl/sr

sigFrames = librosa.util.frame(y, frame_length = winSize, hop_length = hl) 
sigSTE = np.sum(np.square(sigFrames),axis = 0)
mean = np.mean(sigSTE)
x =sigSTE>mean
print(x.shape)
speech = sigSTE[x]
npnspeech = sigSTE[~x]
speechIndices = np.where(speech)
nonspeechIndices = np.where(npnspeech)
vframes = np.array(sigFrames.flatten()[np.where(x==1)], dtype=y.dtype)
print(vframes.shape)
print("Speech Frames: ", str(speechIndices[0].shape[0]))
print("Non Speech Frames: ", str(nonspeechIndices[0].shape[0]))
vad = max(y)*x

d = np.zeros(fr)
duration = 0
f=open('file','+w')
for i in x:
    if(i == True):
        d =d+1
        f.write("s")
        f.write("")
        f.write("{} {}\n" .format(duration,duration+hop_sec))
        duration = duration+hop_sec
        #print(duration)
    else:
        d =d+1
        f.write("nsp")
        f.write("")
        f.write("{} {}\n" .format(duration,duration+hop_sec))
        duration = duration+hop_sec
        
print(d)
        #print(duration) 
f.close()
'''
plot.plot(y)
plot.show()
plot.plot(x)
plot.show()
'''










