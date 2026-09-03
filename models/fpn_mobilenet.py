import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .mobilenet_v2 import MobileNetV2

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super(FPNHead, self).__init__()
        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.block0(x), inplace=True)
        x = F.relu(self.block1(x), inplace=True)
        return x

class FPN(nn.Module):
    """
    Official DeblurGAN-v2 Feature Pyramid Network (FPN) with MobileNetV2 backbone.
    Extracts multi-scale features across 5 pyramidal resolutions with lateral 1x1 convs.
    """
    def __init__(self, norm_layer=None, num_filters=128):
        super(FPN, self).__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)

        net = MobileNetV2(n_class=1000)
        self.features = net.features

        # Backbone hierarchy matching DeblurGAN-v2 key names exactly
        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])

        # Top-down pathway
        self.td1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(inplace=True)
        )
        self.td2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(inplace=True)
        )
        self.td3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(inplace=True)
        )

        # Lateral 1x1 convolutions
        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        map4 = lateral4
        map3 = self.td1(lateral3 + F.interpolate(map4, scale_factor=2, mode='nearest'))
        map2 = self.td2(lateral2 + F.interpolate(map3, scale_factor=2, mode='nearest'))
        map1 = self.td3(lateral1 + F.interpolate(map2, scale_factor=2, mode='nearest'))
        return lateral0, map1, map2, map3, map4

# Compatibility alias
FPNMobileNet = FPN

