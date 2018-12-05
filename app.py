import urllib
import scipy.io.wavfile
import pydub
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft as fft


#Load Env Variables
load_dotenv()

#Path for Audio files
mp3_file_path = os.getenv("audio_folder") + os.getenv("audio_file")
wav_file_path = mp3_file_path.replace("mp3","wav")

#read mp3 file
mp3 = pydub.AudioSegment.from_mp3(mp3_file_path)

#convert to wav
if (os.path.isfile(wav_file_path)):
  print("WAV file already exist. \nDeleting old file...")
  os.remove(wav_file_path)
else:
  print("WAV file not found.")
print("Generating WAV file...")
mp3.export(wav_file_path, format="wav")

#read wav file
rate,audData=scipy.io.wavfile.read(wav_file_path)

#Total samples divided by sampling rate, gives total duration of song
total_duration = audData.shape[0] / rate
total_channels = audData.shape[1]

#Generic Case of 2 channels. Collecting their respective data
channel1=audData[:,0] #left
channel2=audData[:,1] #right

#save wav file
# scipy.io.wavfile.write(temp_folder+"file2.wav", rate, audData)
#save a file at half and double speed
# scipy.io.wavfile.write(wav_file_path.replace("track1","track1a"), int(rate/2), audData)
# scipy.io.wavfile.write(wav_file_path.replace("track1","track2a"), int(rate*2), audData)
#save a single channel
# scipy.io.wavfile.write(wav_file_path.replace("track1","track3a"), rate, channel1)
#averaging the channels damages the music
# mono=np.sum(audData.astype(float), axis=1)/2
# scipy.io.wavfile.write(wav_file_path.replace("track1","track4a"), rate, mono)
#Energy of music
energy = np.sum(channel1.astype(float)**2)
power = 1.0/(2*(channel1.size)+1)*np.sum(channel1.astype(float)**2)/rate
print("Sampling Rate: {} KHz".format(rate))
print("Total Duration of song: {} Seconds".format(total_duration))
print("Total Channels : ",total_channels)
print("Energy :",energy)
print("Power : ", power)

#create a time variable in seconds
time = np.arange(0, float(audData.shape[0]), 1) / rate

#plot amplitude (or loudness) over time
plt.figure(1)
plt.subplot(211)
plt.plot(time, channel1, linewidth=0.01, alpha=0.7, color='#000000')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(212)
plt.plot(time, channel2, linewidth=0.01, alpha=0.7, color='#000000')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

fourier=fft.fft(channel1)

plt.plot(fourier, color='#ff7f00')
plt.xlabel('k')
plt.ylabel('Amplitude')
n = len(channel1)
fourier = fourier[0:int(n/2)]
plt.show()

# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(n)

#calculate the frequency at each point in Hz
freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n);

plt.plot(freqArray/1000, 10*np.log10(fourier), color='#ff7f00', linewidth=0.02)
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.show()

plt.figure(2, figsize=(8,6))
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(channel1, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
plt.show()
np.where(freqs==10034.47265625)
MHZ10=Pxx[233,:]
plt.plot(bins, MHZ10, color='#ff7f00')