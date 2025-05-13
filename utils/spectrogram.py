import torch
import torchaudio
import torch.nn.functional as F

def waveform_to_spectrogram(waveform, n_fft=512, hop_length=128, win_length=None, log=True, eps=1e-8):
    """
    Convert waveform(1D tensor) to (log-)magnitude spectrogram

    Args:
        waveform: shape(T, )
        n_fft: Number of fft bins. Controls frequency resolution.
        hop_length: Step size between windows. Controls time resolution.
        win_length: Actual window size. Defaults to n_fft.
        log: Whether to return log-magnitude spectrogram.
        eps: Small constant epsilon to avoid log(0) when log=True
    
    Returns: Spectrogram of shape(freq_bins, time_frames)
    """
    spectrogram = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length or n_fft,
        return_complex=True
    )
    
    magnitude_spectrogram = spectrogram.abs()
    if log:
        magnitude_spectrogram = torch.log(magnitude_spectrogram + eps)
    
    return magnitude_spectrogram

def spectrogram_to_waveform(magnitude_spectrogram, phase=None, n_fft=512, hop_length=128, win_length=None, num_iters=32, log=True):
    """
    Convert (log-)magnitude spectrogram back to waveform

    Args:
        phase: Optional phase info. If None, uses Griffin-Lim as default.
        num_iters: Number of Griffin-Lim iterations for phase recovery process
    
    Returns: Reconstructed waveform
    """
    if log:
        magnitude_spectrogram = torch.exp(magnitude_spectrogram)
    
    if phase is not None:
        complex_spectrogram = torch.polar(magnitude_spectrogram, phase)
        return torch.istft(
            complex_spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft
        )
    else:
        # Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
        return torchaudio.functional.griffinlim(
            magnitude_spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            power=1.0,
            n_iter=num_iters
        )

def normalize_spectrogram(spec):
    """Min-max normalize a spetrogram to [0,1]"""
    return (spec - spec.min())/(spec.max()-spec.min() + 1e-8)

def denormalize_spectrogram(norm_spec, orig_min, orig_max):
    return norm_spec * (orig_max - orig_min) + orig_min

