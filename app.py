import urllib
import scipy.io.wavfile
import pydub
from dotenv import load_dotenv
import os

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

print(rate)
print(audData)
