# Selective Noise Cancellation (SNC)

A deep learning pipeline for separating meaningful human speech from noisy background environments, inspired by sound separation systems like Spleeter and Demucs.

---

## Overview

This project is a reimplementation of an undergraduate deep learning course project on noise cancellation — now upgraded with:

* A cleaner data pipeline using `torchaudio`
* Torch-native spectrogram operations (no `librosa`)
* Modular UNet and Transformer-based architectures (no pretrained models)
* Reproducible dataset splitting and evaluation

The goal is to isolate meaningful human speech (especially public announcements) from background subway noise using magnitude spectrograms and CNN/Transformer-based models.

---

## Model Architecture

Two models are currently supported:

* **UNet**: A Conv2D-based encoder-decoder baseline
* **AttentionUNet**: UNet with a Transformer bottleneck for richer global context

### AttentionUNet

The AttentionUNet combines a Conv2D-based encoder-decoder structure with a Transformer bottleneck, enabling both local and global pattern learning.

| Stage     | Shape                               | Notes                              |
| --------- | ----------------------------------- | ---------------------------------- |
| Input     | (B, 1, 257, 625)                    | Log-magnitude spectrogram          |
| Encoder   | (B, 64→256, ↓257→32, ↓625→78)       | 3× DoubleConv + strided Conv2d     |
| Attention | (B, 256, 32, 78) → (B, 512, 32, 78) | Flatten → Transformer → DoubleConv |
| Decoder   | (B, 512→64, ↑32→257, ↑78→625)       | 3× UpConv + skip connections       |
| Output    | (B, 1, 257, 625)                    | Predicted clean spectrogram        |

*See [`models/attention_unet.py`](./models/attention_unet.py) for full details.*


## Tensor Shape Convention

This project uses the following standard for spectrogram tensor shapes throughout the pipeline:

| Tensor Type           | Shape        | Description                             |
| --------------------- | ------------ | --------------------------------------- |
| Raw waveform          | (T,)         | 1D time-domain audio                    |
| Spectrogram           | (1, F, T)    | 1-channel magnitude spectrogram         |
| Model input (batched) | (B, 1, F, T) | Batch size B, 1 channel, F freq, T time |
| Model output          | (B, 1, F, T) | Same shape as input                     |
| Reconstructed audio   | (T,)         | From inverse STFT                       |

Where:

* `F` = number of frequency bins (e.g., 257 when n\_fft=512)
* `T` = number of time frames (varies with audio duration and hop size)
* `1` = single audio channel

This shape is consistent across data loading, spectrogram generation, and CNN-based model inputs.

---

## Training Procedure

### 1. Dataset Overview

* **Target (Clean Speech)**:

  * Subway announcements collected from official metro websites (e.g., Seoul Metro).
* **Noise Sources**:

  * **FSD50K clips**: 2 per mixture (filtered by label)
  * **Miscellaneous noise**: 1 clip per mixture (Freesound, subway ambiance, etc.)

### 2. Mixture Generation

* Each mixture is generated **on-the-fly** during training.
* A total of **3 noise clips** are mixed with 1 clean clip.
* **Signal-to-Noise Ratio (SNR)**:

  * Randomized between **-5 to 10 dB** (configurable)
* **Normalization**:

  * RMS normalization applied before mixing for consistency.

### 3. Spectrogram Processing

* All waveforms are **5 seconds** long and mono.
* Transformed into **magnitude spectrograms** using `torch.stft`.
* Spectrogram shape: `(1, 257, 625)`:

  * `1`: single-channel
  * `257`: frequency bins (n\_fft=512)
  * `625`: time frames

### 4. Dataset Splitting

* Dataset is split into:

  * **Train**
  * **Validation**
  * **Test**
* Uses `torch.utils.data.random_split` with a fixed random seed (`torch.Generator(seed=42)`) for reproducibility.

### 5. Model Training

* **Supported Models**: `UNet`, `AttentionUNet`
* **Loss Function**: Mean Squared Error (`nn.MSELoss`)
* **Optimizer Options**:

  * `Adam`, `AdamW`, `SGD` (configurable)
* **Scheduler**:

  * `ReduceLROnPlateau` with patience and decay
* Model learns to predict **clean spectrograms** from noisy mixtures.

### 6. Evaluation Metrics

* **Training/Validation Metric**: Signal-to-Noise Ratio (SNR in dB)
* **Test Metric**: DNSMOS (Deep Noise Suppression Mean Opinion Score)

  * Includes three dimensions:

    * **Speech quality (SIG)**: How clear and natural the speech is.
    * **Background noise intrusiveness (BAK)**: How much the background interferes.
    * **Overall quality (OVR)**: General perceived quality.
  * Higher scores indicate better performance.
  * DNSMOS GitHub: [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)

---

## Results

### Quantitative Metrics

| Model         | Epochs | Val Loss | Val SNR (dB) | DNSMOS Speech | DNSMOS BAK | DNSMOS OVR |
| ------------- | ------ | -------- | ------------ | ------------- | ---------- | ---------- |
| AttentionUNet | 5      | \~0.22   | \~9.3        | **2.59**      | **2.24**   | **1.97**   |

*(Results on 783 samples using current model checkpoint)*

* [ ] Add more DNSMOS results here as training progresses
* [ ] Add test on UNet for comparison

### Audio Examples

TBA: This section will include links to `mixture`, `prediction`, and optionally `clean` samples (as needed for privacy or relevance).

---

## How to Use

### Evaluate a Trained Model

Run the following command to evaluate on the test set:

```bash
python evaluate.py --model attention --checkpoint checkpoints/attention_best_checkpoint.pt --save_audio
```

### Train your Own Model

```bash
python train.py --model attention --epochs 10 --batch_size 8
```

Additional CLI flags (e.g., --optimizer, --snr\_min, --save\_dir) can be found via: train.py

---

## Acknowledgements

This project was inspired by:

* [Spleeter](https://github.com/deezer/spleeter) by Deezer
* [Torchaudio](https://pytorch.org/audio/) and its speech enhancement tutorials
* The original course project at Yonsei University where this work began
* [DNSMOS (Microsoft)](https://github.com/microsoft/DNS-Challenge) — for perceptual quality evaluation of speech enhancement models

Thanks to the maintainers of [FSD50K](https://github.com/eduardofv/FSD50K) and Freesound for providing open-source audio datasets.
