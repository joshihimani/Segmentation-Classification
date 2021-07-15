import librosa
import numpy as np
import matplotlib.pyplot as plot
import librosa.display
y, sr = librosa.load('should.wav')

size = len(y)
hop_length = 256
frame_length = 512

    
energy = np.array([
    sum(abs(y[i:i+frame_length]**2))
    for i in range(0, size, hop_length)
])
a = energy.shape
print(energy)
print(a)


plot.subplot(211)

plot.plot(energy)

plot.xlabel('time')

plot.ylabel('amplitude')
'''

plt.figure()
plt.subplot(1, 1, 1)
librosa.display.waveplot(y, sr=sr) #displays the time domain for the audio signal
#librosa.display.waveplot(energy, sr = sr) # display time domain for the short time energy



# to plot the spectrogram
D = np.abs(librosa.stft(energy))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),y_axis='log', x_axis='time')
plt.title('Spectrogram')
'''