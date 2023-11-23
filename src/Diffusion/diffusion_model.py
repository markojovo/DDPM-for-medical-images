import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)
        x = self.se(x)
        return x

class DecoderBlock(nn.Module):
    # Note the change in the in_channels parameter to account for the skip connections
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels + skip_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)
        x = self.se(x)
        return x

class StepwiseReverseDiffusionNet(nn.Module):
    def __init__(self):
        super(StepwiseReverseDiffusionNet, self).__init__()

        # Encoder blocks
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Decoder
        # Adjust the in_channels to include the skip connections
        self.dec1 = DecoderBlock(512, 256)  # No skip connection for the first decoder block
        self.dec2 = DecoderBlock(256, 128, skip_channels=256)  # Skip connection from enc3 (256 channels)
        self.dec3 = DecoderBlock(128, 64, skip_channels=128)  # Skip connection from enc2 (128 channels)
        self.dec4 = DecoderBlock(64, 1, skip_channels=64)  # Skip connection from enc1 (64 channels)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Decoder with skip connections
        dec1 = self.dec1(enc4)
        dec1 = torch.cat((dec1, enc3), 1)  # Skip connection from enc3
        dec2 = self.dec2(dec1)
        dec2 = torch.cat((dec2, enc2), 1)  # Skip connection from enc2
        dec3 = self.dec3(dec2)
        dec3 = torch.cat((dec3, enc1), 1)  # Skip connection from enc1
        dec4 = self.dec4(dec3)
        
        # Final Activation
        out = torch.sigmoid(dec4)
        return out
