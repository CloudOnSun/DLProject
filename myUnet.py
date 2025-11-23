import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    Simple ResNet-style block:
    x -> ConvBNReLU -> ConvBN -> (add skip) -> ReLU
    If in_channels != out_channels or stride > 1, we use a 1x1 conv in the skip path.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class SmallResUNet(nn.Module):
    """
    Lightweight U-Net with residual blocks.
    - Fully convolutional (no linear layers).
    - Works for arbitrary H,W divisible by 16 (due to 4x downsampling / upsampling).
    """
    def __init__(self, in_channels=3, num_classes=1, base_channels=32):
        super().__init__()

        # -------- Encoder --------
        # Level 1: H, W
        self.enc1 = ResidualBlock(in_channels, base_channels)        # C = 32
        self.pool1 = nn.MaxPool2d(2)  # H/2, W/2

        # Level 2: H/2, W/2
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)  # C = 64
        self.pool2 = nn.MaxPool2d(2)  # H/4, W/4

        # Level 3: H/4, W/4
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)  # C = 128
        self.pool3 = nn.MaxPool2d(2)  # H/8, W/8

        # Bottleneck: H/8, W/8
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8)  # C = 256

        # -------- Decoder --------
        # Level 3 decoder: up to H/4, W/4
        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4,
            kernel_size=2, stride=2
        )
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4)

        # Level 2 decoder: up to H/2, W/2
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2,
            kernel_size=2, stride=2
        )
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2)

        # Level 1 decoder: up to H, W
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels,
            kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock(base_channels * 2, base_channels)

        # Final 1x1 conv to get per-pixel logits
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # Kaiming (He) initialization for convs with ReLU, as recommended for ReLU activations.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # H,   W,   C=32
        p1 = self.pool1(e1)      # H/2, W/2

        e2 = self.enc2(p1)       # C=64
        p2 = self.pool2(e2)      # H/4, W/4

        e3 = self.enc3(p2)       # C=128
        p3 = self.pool3(e3)      # H/8, W/8

        b = self.bottleneck(p3)  # C=256

        # Decoder
        u3 = self.up3(b)         # H/4, W/4, C=128
        # Pad if needed (in case of odd sizes)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))  # C = 128

        u2 = self.up2(d3)        # H/2, W/2, C=64
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # C = 64

        u1 = self.up1(d2)        # H, W, C=32
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # C = 32

        logits = self.out_conv(d1)  # (N, num_classes, H, W)
        return logits
