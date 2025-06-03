import torch

zero_signal_count = 0
def snr(pred, target, eps=1e-8):
    global zero_signal_count
    signal_power = torch.mean(target**2)
    noise_power = torch.mean((target - pred)**2)
    
    if signal_power < eps:
        zero_signal_count += 1
        return torch.tensor(0.0) # signal is too weak to evaluate SNR
    if noise_power < eps:
        return torch.tensor(float("inf")) # perfect reconstruction
    
    snr = 10 * torch.log10(signal_power / (noise_power + eps))
    return snr