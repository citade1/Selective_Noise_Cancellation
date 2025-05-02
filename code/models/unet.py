import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256]):
        super(UNet, self).__init__()

        # Encoder
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1]*2)

        # Decoder
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2) # this convtranspose doubles spatial size
            )
            self.ups.append(self._block(feature*2, feature))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    # Input -> x:(B, 1, F, T) = (batch_size, channels=1, frequency_bins, time_frames)
    def forward(self, x): 
        
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        
        # Bottleneck
        x = self.bottlenect(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # ConvTranspose2d 
            skip_connections = skip_connections[i//2]
            if x.shape != skip_connections.shape:
                x = F.interpolate(x, size=skip_connections.shape[2:]) # Match size
            x = torch.cat((skip_connections, x), dim=1) # channel-wise concat
            x = self.ups[i+1](x) # Conv block
        
        return self.final_conv(x)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(replace=True)
        )
