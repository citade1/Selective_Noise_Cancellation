import os
import torch
import random
from torch.utils.data import Dataset
from utils import mix_audio, waveform_to_spectrogram

# ---------------------
# Custom Dataset
# ---------------------
class SpectrogramDataset(Dataset):
    def __init__(self, target_dir, noise_fsd_dir, noise_misc_dir , snr_range=(5,15), n_noises=3, mode="clean"):
        self.target_paths = self._gather_paths(target_dir)
        self.noise_fsd_paths = self._gather_paths(noise_fsd_dir)
        self.noise_misc_paths = self._gather_paths(noise_misc_dir)
        self.snr_range = snr_range
        self.n_noises = n_noises
        self.mode = mode
    
    def _gather_paths(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]

    def __len__(self):
        return len(self.target_paths)
    
    def __getitem__(self, idx):
        target_waveform = torch.load(self.target_paths[idx])
        target_waveform = target_waveform.squeeze()  # safely reduce to 1D 

        # Load and normalize noise waveforms
        fsd_noises = random.sample(self.noise_fsd_paths, self.n_noises - 1)
        misc_noise = random.choice(self.noise_misc_paths)
        noise_paths = fsd_noises + [misc_noise]

        noise_waveforms = []
        for path in noise_paths:
            noise = torch.load(path).squeeze() # reduce to 1D 
            noise_waveforms.append(noise)

        mixed_noise = sum(noise_waveforms) / len(noise_waveforms)

        # Mix target + noise
        snr_db = random.uniform(*self.snr_range)
        mixture, target = mix_audio(target_waveform, mixed_noise, snr_db, mode=self.mode) # if mode is "clean" the return pair will be (mixture, clean), otherwise (mixture, noise)
        # get phase and waveform length from mixture 
        phase = torch.stft(mixture, n_fft=512, hop_length=128, return_complex=True).angle() # extract phase from mixture
        length = mixture.size(-1)

        # Spectrograms: (1, F, T)
        mixture_spec = waveform_to_spectrogram(mixture.squeeze()) # expect 1D input -> squeeze
        target_spec = waveform_to_spectrogram(target.squeeze())

        return mixture_spec, target_spec, phase, length