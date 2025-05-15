import os
import torch
import random
from torch.utils.data import Dataset
from utils import mix_audio, waveform_to_spectrogram

# ---------------------
# Custom Dataset
# ---------------------
class SpectrogramDataset(Dataset):
    def __init__(self, target_dir, noise_fsd_dir, noise_misc_dir , snr_range=(5,15), n_noises=3):
        self.target_paths = self._gather_paths(target_dir)
        self.noise_fsd_paths = self._gather_paths(noise_fsd_dir)
        self.noise_misc_paths = self._gather_paths(noise_misc_dir)
        self.snr_range = snr_range
        self.n_noises = n_noises
    
    def _gather_paths(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]

    def __len__(self):
        return len(self.target_paths)
    
    def __getitem__(self, idx):
        # load target
        target_waveform = torch.load(self.target_paths[idx])
        
        # load 2 FSD50K + 1 miscellaneous noise
        fsd_noises = random.sample(self.noise_fsd_paths, 2)
        misc_noise = random.choice(self.noise_misc_paths)
        noise_path = fsd_noises + [misc_noise]
        noise_waveform = [torch.load(p) for p in noise_path]
        mixed_noise = sum(noise_waveform) / len(noise_waveform)

        # mix and return
        snr_db = random.uniform(*self.snr_range)
        mixture, clean = mix_audio(target_waveform, mixed_noise, snr_db)
        return waveform_to_spectrogram(mixture), waveform_to_spectrogram(clean)
