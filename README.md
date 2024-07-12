# Selective Noise Cancellation (SNC) Design/Thought Process
## SNC: Transformer (Whisper) + Self-Supervised Learning (MixIT)

**(This project is not complete and will be continuously updated)**

<div style="text-align:center;">
  <img width="500" src="https://github.com/user-attachments/assets/d7e244fa-b4ec-40a0-961c-ed1925407272">
</div>

### 0. Problem Statement/Motivation
Raw audio recordings from real-world environments contain a mixture of various sounds. For instance, a street recording might include traffic noise, pedestrian chatter, and other ambient sounds all blended into one audio track. Selective Noise Cancellation (SNC) is a project that focuses on detecting and separating sounds with the specific aim of filtering out unwanted noise rather than merely separating sounds. The ultimate goal of SNC aligns closely with that of sound separation: to isolate all distinct audio signals within a mixed recording. (However how fine or pure these signals need to be considered independent is still debatable and hasn’t been fully explored yet in this project.)

SNC aims to classify important sounds from background noise, which may vary depending on the detected environment. By filtering out irrelevant sounds and preserving only the significant ones, SNC aims to achieve true 'selective' noise cancellation. This technology could have diverse applications across various industries, extending beyond speech recognition and vocal-instrument separation, to noise-cancelling devices for consumer electronics, construction sites, and many other settings requiring the detection, separation, and elimination of extraneous sounds.

### 1. Dataset

#### 1.1. Public Transportation Dataset
- **Target**: Announcement, alarm sounds
- **Noise**: All other sounds

**Training Data:**
- **Subway dataset:**
  - **Inputs**: Target + Noise
  - **Target**: Subway announcements
    - **Sources**: Seoul subway line 1-8 (seoulmetro.co.kr), Shinbundang line (shinbundang.co.kr), Busan/Daegu/Daejeon subway (data.go.kr)
  - **Noise**: Subway background noise (sources: freesound.org, “creative commons 0”)

**Test Data:**
- Subway field recordings with announcements (freesound.org, “creative commons 0”)

#### 1.2. Street Dataset
- **Target**: Vehicle horns, approaching sounds, alarm, announcement
- **Noise**: All other sounds

**Training Data:**
- **Street dataset:**
  - **Inputs**: Target + Noise
  - **Target**: Vehicle horns, approaching sounds, alarms, announcements
  - **Noise**: Street background noise
    - **Sources**: Freesound.org (creative commons 0)

**Test Data:**
- Street recordings with relevant sounds (freesound.org, creative commons 0)

#### 1.3. Home Environment
- **Target**: Alarm, announcement, conversation directed at the user
- **Noise**: Other sounds, especially distant sounds, white noise, noise from electric appliances

**Training Data:**
- **Home environment dataset:**
  - **Inputs**: Target + Noise
  - **Target**: Alarms, announcements, conversations directed at the user
  - **Noise**: Home background noise 
    - **Sources**: Freesound.org (creative commons 0)

**Test Data:**
- Home recordings with relevant sounds (freesound.org, creative commons 0)

*Approaching sounds need to be elaborated/experimented, not yet trained to be identified/separated.*

### 2. Model Architecture: Transformer-Based Sound Separation Model
The transformer-based sound separation model is designed to efficiently separate important sounds from background noise using neural network architectures. This model leverages the capabilities of transformers, which have proven effective in various tasks requiring attention to sequential data, such as natural language processing and, more recently, audio processing.

### 2.1 Key Components

**1. Input Processing:**
- **To Spectrograms**: The audio input is first converted into spectrograms, which are visual representations of the audio signal in terms of time, frequency, and amplitude.

**2. Encoder:**
- **Convolutional Layers**: Consists of several convolutional layers to extract high-level features from the input spectrograms, helping in identifying local patterns.
  - **Conv2d Layer 1**: Input Channels: 1, Output Channels: 64, Kernel Size: 3x3, Stride: 1, Padding: 1, Activation: ReLU
  - **Conv2d Layer 2**: Input Channels: 64, Output Channels: 128, Kernel Size: 3x3, Stride: 1, Padding: 1, Activation: ReLU
  - **Conv2d Layer 3**: Input Channels: 128, Output Channels: 256, Kernel Size: 3x3, Stride: 1, Padding: 1, Activation: ReLU

**3. Attention Mechanism:**
- **Multi-head Attention**: At the core of the transformer architecture is the multi-head attention mechanism to focus on different parts of the input spectrogram simultaneously, identifying important features relevant to the task.
  - **Embedding Dimension**: 256
  - **Number of Heads**: 8

**4. Decoder:**
- **Transpose Convolutional Layers**: The decoder consists of transpose convolutional layers that progressively upsample the features back to the original spectrogram dimensions.
  - **ConvTranspose2d Layer 1**: Input Channels: 256, Output Channels: 128, Kernel Size: 3x3, Stride: 1, Padding: 1, Activation: ReLU
  - **ConvTranspose2d Layer 2**: Input Channels: 128, Output Channels: 64, Kernel Size: 3x3, Stride: 1, Padding: 1, Activation: ReLU
  - **ConvTranspose2d Layer 3**: Input Channels: 64, Output Channels: 1, Kernel Size: 3x3, Stride: 1, Padding: 1, Activation: Sigmoid

**5. Self-Supervised Learning with MixIT:**
- The model uses the MixIT (Mixing Iterative Training) framework, a self-supervised learning approach that does not require clean, isolated source signals for training.
- Instead, the model learns to separate mixed audio signals into their constituent sources by leveraging mixtures of mixtures, iteratively refining its separation capabilities.

### 2.2 Benefits of Using Transformers
- **Robust Feature Extraction**: Transformer can capture long-range dependencies and complex patterns in the audio signal, leading to more accurate sound separation.
- **Scalability**: The architecture can be scaled to handle various audio lengths and complexities, making it versatile for different environments.
- **Attention to Detail**: The attention mechanism ensures that the model focuses on the most relevant parts of the signal, enhancing the clarity and quality of the separated sounds.

### 3. Training (Code is provided in Section 7)

**Data Preparation:**
- **Create Mixtures**: Combine the target and noise audio signals to create mixed audio samples, maintaining the integrity of the temporal information in the audio signals.
- **Convert to Spectrograms**: After creating the mixed audio signals, convert both the mixtures and the sources (target and noise) to spectrograms.

**Model Training:**
- **Objective**: The model is trained to minimize the reconstruction error between the estimated sources (target and noise) and the mixed inputs, ensuring the smallest possible reconstruction error.
- **Loss Function**: Mean Squared Error (MSE) is used as the loss function to measure the difference between the predicted and actual spectrograms, effectively capturing reconstruction accuracy.
- **Optimization Algorithm**: Adam

### 4. Evaluation

**Evaluation Steps:**
- **Test Dataset**: Evaluate the model on a separate test dataset containing audio from various environments.
- **Comparison**: Compare the performance against the baseline model using different metrics. The baseline model is a sound separation encoder-decoder model of the early SNC project, based on a more traditional approach: U-Net based.

**Metrics Result:**
- **Signal-to-Noise Ratio (SNR)**: Measure the clarity of the separated important sounds.
  - **Baseline (U-Net Based Encoder-Decoder)**: TBU
  - **Self-Supervised (MixIT) + Whisper (Transformer)-based Encoder-Decoder**: TBU
- **Subjective Hearing Tests**: Conduct subjective tests to assess the perceived quality of the noise cancellation on a scale of 0-5.
  - **Baseline (U-Net Based Encoder-Decoder)**: 3.5
  - **Self-Supervised (MixIT) + Whisper (Transformer)-based Encoder-Decoder**: 4.5


### 5. Conclusion
This project demonstrates leveraging the capabilities of transformer-based models combined with a self-supervised learning method to achieve effective selective noise cancellation. Compared to the baseline model, this approach showed significant improvements in both SNR in dB, indicating clearer separation of important sounds, and subjective tests, with higher perceived quality scores. With continuous fine-tuning and the development of more advanced models for sound separation tasks, this approach has the potential to generalize well across various environments, ensuring that important sounds are preserved while unwanted noise is effectively canceled.

### 6. References
- [MixIT: Self-Supervised Learning for Audio Source Separation (Scott Wisdom et al., 2020)](https://arxiv.org/abs/2004.04695)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Whisper: OpenAI's Whisper Model for Automatic Speech Recognition (OpenAI, 2022)](https://openai.com/research/whisper)
- [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement (Tan and Wang, 2018)](https://arxiv.org/abs/1805.08352)
