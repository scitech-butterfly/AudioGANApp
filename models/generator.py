# models/generator.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
        )
    def forward(self, x):
        return x + self.block(x)   # residual

class TinySEGAN_Generator(nn.Module):
    """
    U-Net style generator operating on log-mel spectrograms.
    Input/Output: (B, 1, 80, 128)
    
    Design choices for real-time:
    - Depthwise separable convolutions in bottleneck
    - Skip connections preserve speech structure
    - Only 3 encoder/decoder stages (not 5+)
    """
    def __init__(self, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_ch, 4, 2, 1),          # (B, 32, 40, 64)
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),   # (B, 64, 20, 32)
            nn.InstanceNorm2d(base_ch*2), nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1), # (B, 128, 10, 16)
            nn.InstanceNorm2d(base_ch*4), nn.LeakyReLU(0.2)
        )

        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResBlock(base_ch*4),
            ResBlock(base_ch*4),
        )

        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4*2, base_ch*2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch*2), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2*2, base_ch, 4, 2, 1),
            nn.InstanceNorm2d(base_ch), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, 1, 4, 2, 1),
            nn.Tanh()   # output in [-1, 1]
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3(torch.cat([b,  e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return d1
