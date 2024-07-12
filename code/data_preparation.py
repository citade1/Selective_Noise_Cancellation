## Data Preparation

from google.colab import drive
drive.mount('/content/drive')

import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from transformers import WhisperProcessor, WhisperModel

# Paths to be updated 
target_path = '/content/drive/MyDrive/sncWhisper/target'
noise_path = '/content/drive/MyDrive/sncWhisper/noise'

def load_audio_files(file_paths):
    audio_data = []
    for file_path in file_paths:
        audio, sr = librosa.load(file_path, sr=16000)
        audio_data.append(audio)
    return np.array(audio_data)

# Get list of all target and noise audio files
target_files = glob.glob(os.path.join(target_path, '*.wav'))
noise_files = glob.glob(os.path.join(noise_path, '*.wav'))

# Load audio data
target_audios = load_audio_files(target_files)
noise_audios = load_audio_files(noise_files)

# Ensure target and noise arrays have the same shape for mixing
min_len = min(len(target_audios), len(noise_audios))
target_audios = target_audios[:min_len]
noise_audios = noise_audios[:min_len]

# Mix target and noise to create the mixtures
mixtures = target_audios + noise_audios
sources = target_audios

# Convert audio to spectrograms
def audio_to_spectrogram(audio_data):
    spectrograms = []
    for audio in audio_data:
        stft = librosa.stft(audio, n_fft=1024, hop_length=512)
        spectrogram = np.abs(stft)
        spectrograms.append(spectrogram)
    return np.array(spectrograms)

# Convert mixtures and sources to spectrograms
mixture_spectrograms = audio_to_spectrogram(mixtures)
source_spectrograms = audio_to_spectrogram(sources)
