
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(c * 2, c * 2, kernel_size=3, padding=1, groups=c * 2)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(c, c, kernel_size=1)
        self.norm = nn.GroupNorm(1, c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, kernel_size=1))

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv2(x)
        return res + x


class HighResSemiconductorNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_dim=96, scale_factor=2):
        super().__init__()
        self.scale_factor = max(1, scale_factor)
        self.in_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        self.enc1 = nn.Sequential(*[NAFBlock(base_dim) for _ in range(3)])
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=2, stride=2)
        self.enc2 = nn.Sequential(*[NAFBlock(base_dim * 2) for _ in range(3)])
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(*[NAFBlock(base_dim * 4) for _ in range(4)])
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(*[NAFBlock(base_dim * 2) for _ in range(3)])
        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(*[NAFBlock(base_dim) for _ in range(3)])
        self.upsample_head = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * (self.scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale_factor),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, target_hw=None):
        x_in = self.in_conv(x)
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        d2 = self.dec2(self.up2(b) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        out = self.upsample_head(d1)
        if target_hw is not None and out.shape[-2:] != tuple(target_hw):
            out = F.interpolate(out, size=target_hw, mode='bicubic', align_corners=False)
        return out
