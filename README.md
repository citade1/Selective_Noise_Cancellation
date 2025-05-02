# Selective Noise Cancellation(SNC)

## Model Architecture
### Baseline(UNet)
This project uses a baseline UNet for spectrogram-to-spectrogram source separation.

- Input: Noisy spectrogram `(1, F, T)`
- Output: Clean subway announcement spectrogram `(1, F, T)`
- Model: Symmetric encoder-decoder UNet with skip connections

| Stage        | Shape                         | Notes                          |
|--------------|-------------------------------|--------------------------------|
| Input        | (B, 1, 256, 256)              | Input spectrogram              |
| Encoder      | (B, 64→256, ↓256→↓32,↓256→↓32)| 2×Conv per level, downsample   |
| Bottleneck   | (B, 512, 32, 32)              | Deepest layer                  |
| Decoder      | (B, 256→64, ↑32→↑256,↑32→↑256)| Upconv + skip connections      |
| Output       | (B, 1, 256, 256)              | Predicted clean spectrogram    |

_See `code/models/unet.py` for full details._
