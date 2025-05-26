import torchaudio
import torch
import torch.nn.functional as F
import random
import os
from pydub import AudioSegment

DEFAULT_SR = 16000


def load_audio(filepath, sr=DEFAULT_SR):
    """Load an audio file(first with torchaudio, fallback to pydub). Returns 1D waveform"""
    try: 
        waveform, original_sr = torchaudio.load(filepath)
        if original_sr != sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=original_sr, new_freq=sr)
    
    except Exception as e:
        try:
            audio = AudioSegment.from_file(filepath)
            audio = audio.set_channels(1).set_frame_rate(sr)
            samples = audio.get_array_of_samples()
            waveform = torch.tensor(samples, dtpye=torch.float32) / (2**15)
            waveform = waveform.unsqueeze(0)
        except Exception as fallback_error:
            raise RuntimeError(f"Failed to load{filepath} with both torchaudio and pydub. \nOriginal error: {e}\nFallback error:{fallback_error}")
    
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform.squeeze(0), sr
        
def save_audio(filepath, waveform, sr=DEFAULT_SR):
    if waveform.dim()==1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(filepath, waveform, sr)

def match_length(target_wave, noise_wave):
    """Truncate, pad or loop the noise to match target's length"""
    t_len = target_wave.shape[-1]
    n_len = noise_wave.shape[-1]

    if n_len >= t_len:
        start = random.randint(0, n_len-t_len)
        return noise_wave[start: start+t_len]
    else:
        repeat_factor = (t_len // n_len) + 1
        extended = noise_wave.repeat(repeat_factor)[:t_len]
        return extended


def mix_audio(target_wave, noise_wave, snr_db):
    """Mix clean target and noise at a desired SNR level"""
    if target_wave.shape[-1] != noise_wave.shape[-1]:
        noise_wave = match_length(target_wave, noise_wave)

    # ensure both target and noise are mono (1,T)
    if noise_wave.dim()==1:
        noise_wave = noise_wave.unsqueeze(0)
    elif noise_wave.shape[0]>1:
        noise_wave = noise_wave.mean(dim=0, keepdim=True)
    if target_wave.dim()==1:
        target_wave = target_wave.unsqueeze(0)
    elif target_wave.shape[0]>1:
        target_wave = target_wave.mean(dim=0, keepdim=True)

    # scale noise to achieve desired SNR
    signal_power = torch.mean(target_wave**2)
    noise_power = torch.mean(noise_wave**2)
    factor = torch.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power + 1e-8))
    noise_scaled = noise_wave * factor

    mixture = target_wave + noise_scaled
    
    return mixture.clamp(-1.0, 1.0), target_wave


def list_audio_files(directory, exts=(".wav", ".flac")):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(exts)]


def normalize_audio(waveform, method="peak", target_level=0.1):
    """
    Normalize audio waveform.
    - method="peak": scales by max absolute value
    - method="rms": scales to target RMS level
    """
    if method=="peak":
        peak = waveform.abs().max()
        return waveform / peak.clamp(min=1e-8)
    elif method=="rms":
        rms = torch.sqrt(torch.mean(waveform ** 2))
        return waveform * target_level / (rms + 1e-8)
    else:
        raise ValueError("Unknown normalization method. Use 'peak' or 'rms'.")


def mp3_to_wav(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"mp3 file not found : {input_path}")
    audio = AudioSegment.from_mp3(input_path)
    audio.export(output_path, format="wav")


def to_fixed_length(waveform, sr, duration_sec=5.0):
    """
    Splits or pad waveform to fixed-length segments(5 seconds)
    """
    if waveform.dim()==1:
        waveform = waveform.unsqueeze(0) # ensure shape (1,T)
    
    total_len = waveform.shape[-1]
    fixed_len = int(duration_sec * sr)
    
    segments = []
    num_chunks = total_len // fixed_len

    for i in range(num_chunks):
        segment = waveform[:, i*fixed_len:(i+1)*fixed_len]
        segments.append(segment)
        
    
    # handle remainder by padding
    remaining = total_len % fixed_len
    if remaining > 0:
        last = waveform[:, -remaining:]
        pad_length = fixed_len - remaining
        padded = F.pad(last, (0, pad_length))
        segments.append(padded)

    return segments
