import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperModel

class TransformerSoundSeparator(nn.Module):
    def __init__(self):
        super(TransformerSoundSeparator, self).__init__()
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperModel.from_pretrained("openai/whisper-small")
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.processor(x, return_tensors="pt").input_values
        x = self.model(x).last_hidden_state.unsqueeze(1)  # add channel dimension
        x = self.encoder(x)
        x = x.permute(2, 0, 1, 3)  # rearrange for attention layer (L, N, E)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0, 3)  # rearrange back to (N, E, L, C)
        x = self.decoder(x)
        return x
