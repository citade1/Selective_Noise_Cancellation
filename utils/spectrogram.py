import torch
import torchaudio
import torch.nn.functional as F

def waveform_to_spectrogram(waveform, n_fft=512, hop_length=128, win_length=512, log=False, eps=1e-8):
    """
    Convert waveform(T,) to magnitude spectrogram

    Args:
        waveform: shape(T, )
        n_fft: Number of fft bins. Controls frequency resolution.
        hop_length: Step size between windows. Controls time resolution.
        win_length: Actual window size. Defaults to n_fft.
        log: Whether to return log-magnitude spectrogram.
        eps: Small constant epsilon to avoid log(0) when log=True
    
    Returns: torch.Tensor: (1, F, T) single-channel spectrogram for CNN input
    """
    window = torch.hann_window(win_length or n_fft, device=waveform.device)
    spectrogram = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length or n_fft,
        return_complex=True,
        window=window
    )
    magnitude_spectrogram = spectrogram.abs()
    
    if log:
        magnitude_spectrogram = torch.log(magnitude_spectrogram + eps)
    
    return magnitude_spectrogram.unsqueeze(0)

def spectrogram_to_waveform(magnitude_spectrogram, phase=None, length=None, n_fft=512, win_length=512, hop_length=128, num_iters=32, momentum=0.99, log=False):
    """
    Convert magnitude spectrogram (F, T) or (1, F, T) to waveform (T,)
    
    Returns:
        waveform: torch.Tensor, shape (T,)
    """
    if magnitude_spectrogram.dim() == 3:
        magnitude_spectrogram = magnitude_spectrogram.squeeze(0)  # shape (F, T)
    
    if log:
        magnitude_spectrogram = torch.exp(magnitude_spectrogram)


    window = torch.hann_window(win_length, device=magnitude_spectrogram.device)
    if phase is not None:
        # Use the given phase to reconstruct the complex STFT
        complex_spectrogram = torch.polar(magnitude_spectrogram, phase)  # shape (F, T)
        return torch.istft(
            complex_spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=length,
            return_complex=False
        )
    else:
        # Griffin-Lim phase reconstruction
        return torchaudio.functional.griffinlim(
            magnitude_spectrogram,
            window=window,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
            n_iter=num_iters,
            momentum=momentum,
            length=length,
            rand_init=True
        )

def normalize_spectrogram(spec):
    """Min-max normalize a spetrogram to [0,1]"""
    return (spec - spec.min())/(spec.max()-spec.min() + 1e-8)

def denormalize_spectrogram(norm_spec, orig_min, orig_max):
    return norm_spec * (orig_max - orig_min) + orig_min

