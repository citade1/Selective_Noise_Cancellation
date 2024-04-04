
# Produce Spectrogram for audio

import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import skimage.io
from pydub import AudioSegment
import os
import math

# defining data path
root = "aeDataset"
audio_list = ['1123 신분당선 신사-상현.m4a', '1123 3호선 종로3가-신사.m4a', '9호선 급행 당산-고속터미널 11월 23일.m4a', '6호선 합정-약수 11월 25일.m4a', '3호선 을지로3가-교대 11월 24일.m4a', '2호선 신촌-을지로3가 11월 24일.m4a', '2호선 신촌-당산 11월 23일.m4a']

# convert m4a into wav
for i in audio_list:
  if i != 'train' and i!='test' and i!='wav':
    track = AudioSegment.from_file(root+i, format = 'm4a')
    handle = track.export(root+'wav/'+i+'_to wav', format='wav')


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

# audio split function
class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + filename
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, sec_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+sec_per_split, split_fn)
            print(str(i) + ' Done')

# split audio file into 10s and transform it into spectrogram
def split_to_spectrogram(in_path, out_path):
    # split wav file into 10 seconds chunk 
    folder = root+in_path 
    wav_list = os.listdir(folder)
    for i in wav_list:
        file = i
        split_wav = SplitWavAudioMubin(folder, file)
        split_wav.multiple_split(sec_per_split=10)

    # change wav file into spectrogram...
    split_wav_list = os.listdir(folder)
    for i in split_wav_list:

        audio = folder + i
        ipd.Audio(audio, rate=sample_rate)
        # load audio files with librosa
        y, sample_rate = librosa.load(audio)

        hop_length = 256
        n_fft = 1024

        mel_spectrogram = librosa.feature.melspectrogram(y, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
        # min-max scale to fit inside 8-bit range
        img = scale_minmax(log_mel_spectrogram, 0, 255).astype(np.uint8)
    
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255-img # invert. make black==more energy
    
        save_path = root+out_path
        skimage.io.imsave(save_path+i+'.png', img) # save image into size 128 x 431

split_to_spectrogram('wav/', 'train/subway/')
split_to_spectrogram('wavother','train/other/')
split_to_spectrogram('wavtest','test/subway/')
split_to_spectrogram('wavothertest','test/other/')

