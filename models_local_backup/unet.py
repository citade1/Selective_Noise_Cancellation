import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256]):
        super(UNet, self).__init__()

        # Encoder: Conv -> BN -> ReLU -> Strided Conv for downsampling
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for feature in features:
            self.encoder_blocks.append(self._conv_block(in_channels, feature))
            self.downsamples.append(nn.Conv2d(feature, feature, kernel_size=2, stride=2))
            in_channels=feature

        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1]*2)

        # Decoder: Upsample -> Concat(w/ skip connection) -> Conv(->BN ->ReLU)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = features[::-1]
        
        for feature in reversed_features:
            self.upsamples.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder_blocks.append(self._conv_block(feature*2, feature))
        
        # Final 1x1 conv for mask output(0~1)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder, downsample in zip(self.encoder_blocks, self.downsamples):
            x = encoder(x)
            skip_connections.append(x)
            x = downsample(x) 
        
        # Bottleneck 
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for i in range(len(self.upsamples)):
            x = self.upsamples[i](x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_blocks[i](x)
        
        x = self.final_conv(x)
        return x


    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )