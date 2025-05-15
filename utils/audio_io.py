import torchaudio
import torch
import random
import os
from pydub import AudioSegment

DEFAULT_SR = 16000


def load_audio(filepath, sr=DEFAULT_SR):
    """Load an audio file and resample if needed"""
    waveform, original_sr = torchaudio.load(filepath)
    if original_sr != sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=original_sr, new_freq=sr)
    return waveform.squeeze(0), sr


def save_audio(filepath, waveform, sr=DEFAULT_SR):
    """Save a waveform to disk"""
    torchaudio.save(filepath, waveform.unsqueeze(0), sr)


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
    Splits a waveform into segments of fixed length (default = 5 seconds)
    Pads the last segment by repeating if it's shorter than the desired length
    """
    if waveform.dim()==2:
        waveform = waveform.squeeze(0) # assume mono channel 
    
    split_waveforms = []
    total_len = waveform.shape[-1]
    waveform_len = duration_sec * sr
    start = 0

    while start + waveform_len <= total_len:
        split_waveforms.append(waveform[start:start+waveform_len])
        start += waveform_len
    
    # handle remainder by padding
    remaining = waveform[start:]
    if remaining.shape[-1] > 0:
        repeat_factor = (waveform_len // remaining.shape[-1]) + 1
        padded = remaining.repeat(repeat_factor)[:waveform_len]
        split_waveforms.append(padded)

    return split_waveforms
