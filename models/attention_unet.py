import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.double_conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256], attention_block=None, attn_kwargs=None):
        super().__init__()

        # Encoder block
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            self.downsamples.append(nn.Conv2d(feature, feature, kernel_size=2, stride=2))
            in_channels = feature
        
        # Bottleneck Attention (optional, pluggable)
        self.use_attention = attention_block is not None
        if self.use_attention:
            self.attn = attention_block(**attn_kwargs) 
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Decoder block
        self.upsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for feature in reversed(features):
            self.upsamples.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature*2, feature))

    
        self.final_conv = nn.Conv2d(feature, out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc, down in zip(self.encoder, self.downsamples):
            x = enc(x)
            skip_connections.append(x)
            x = down(x)
        
        skip_connections = skip_connections[::-1]
        
        # Bottleneck
        if self.use_attention:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().reshape(B, H*W, C) # shape = (B, T, C), H*W becomes the sequence length T, and channel_num C becomes feature dimension
            x = self.attn(x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() # Back to (B, C, H, W)
        
        # always use this bottleneck (with or without attention) to match the shape and contain expressiveness    
        x = self.bottleneck(x)
        
        # Decoder
        for idx in range(0, len(self.upsamples)):
            x = self.upsamples[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear")
            x = torch.concat((x, skip_connection), dim=1)
            x = self.decoder[idx](x)

        return self.final_conv(x)