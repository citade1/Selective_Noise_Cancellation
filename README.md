# Selective Noise Cancellation (SNC)
A deep learning pipeline for separating meaningful human speech from noisy background environments, inspired by sound separation systems like Spleeter and Demucs.

---
## Overview

This project is a reimplementation of an undergraduate deep learning course project on noise cancellation — now upgraded with:

- A cleaner data pipeline using `torchaudio`
- Torch-native spectrogram operations (no `librosa`)
- Modular UNet and Transformer-based architectures (no pretrained models)
- Reproducible dataset splitting and evaluation

The goal is to isolate meaningful human speech (especially public announcements) from background subway noise using magnitude spectrograms and CNN/Transformer-based models.

---
## Model Architecture

Two models are currently supported:

- **UNet**: A Conv2D-based encoder-decoder baseline
- **AttentionUNet**: UNet with a Transformer bottleneck for richer global context

### AttentionUNet 

The AttentionUNet combines a Conv2D-based encoder-decoder structure with a Transformer bottleneck, enabling both local and global pattern learning.

| Stage     | Shape                               | Notes                                |
|-----------|-------------------------------------|--------------------------------------|
| Input     | (B, 1, 257, 625)                    | Log-magnitude spectrogram            |
| Encoder   | (B, 64→256, ↓257→32, ↓625→78)       | 3× DoubleConv + strided Conv2d       |
| Attention | (B, 256, 32, 78) → (B, 512, 32, 78) | Flatten → Transformer → DoubleConv  |
| Decoder   | (B, 512→64, ↑32→257, ↑78→625)       | 3× UpConv + skip connections         |
| Output    | (B, 1, 257, 625)                    | Predicted clean spectrogram          |

_See [`models/attention_unet.py`](./models/attention_unet.py) for full details._

---
## Training Procedure

### Dataset
- **Target (Clean)**: Subway announcements sourced from official metro websites (e.g., Seoul Metro).
- **Noise Sources**:
  - **FSD50K**: 2 ambient/environmental clips per mixture, filtered by label
  - **Miscellaneous**: 1 additional noise clip from Freesound or custom recordings

### On-the-Fly Mixture Generation
- Clean + 3 noise clips are mixed dynamically during training
- Each mixture uses a random Signal-to-Noise Ratio (SNR) between **5–15 dB**
- RMS normalization is applied to standardize mixing

### Data Splitting
- Dataset is split into **train**, **validation**, and **test** sets
- Split is done with `torch.utils.data.random_split` using a fixed `torch.Generator(seed=42)` for reproducibility

### Spectrogram Processing
- All audio clips are **5 seconds long**
- Each waveform is converted to a **log-magnitude spectrogram** using `torch.stft`
- Final input shape: `(1, 257, 625)` — (channel, frequency bins, time frames)

### Training
- Supported models: `UNet`, `AttentionUNet`
- Optimizer: `Adam`, `AdamW`, or `SGD` (configurable via CLI)
- The model predicts a clean spectrogram from noisy input mixtures

### Evaluation Metrics
- **Loss Function**: Mean Squared Error (`torch.nn.MSELoss`)
- **Primary Metric**: Signal-to-Noise Ratio (SNR), reported in decibels

---
## Results

Quantitative results will be added shortly.
Planned:
Signal-to-Noise Ratio (SNR)
(Optional) PESQ and STOI for perceptual quality

Audio samples of (mixture, clean, noise, predicted) will be shared.

---
## How to Use

### Evaluate a Trained Model
Run the following command to evaluate on the test set:

```bash
python evaluate.py --model attention --checkpoint checkpoints/attention_best.pt --save_audio
```
### Train your Own Model
```bash
python train.py --model attention --epochs 10 --batch_size 8
```
Additional CLI flags (e.g., --optimizer, --snr_min, --save_dir) can be found via: train.py

---
## Acknowledgements

This project was inspired by:
- [Spleeter](https://github.com/deezer/spleeter) by Deezer
- [Torchaudio](https://pytorch.org/audio/) and its speech enhancement tutorials
- The original course project at Yonsei University where this work began 

Thanks to the maintainers of [FSD50K](https://github.com/eduardofv/FSD50K) and Freesound for providing open-source audio datasets.
