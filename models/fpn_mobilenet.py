import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class FPNMobileNet(nn.Module):
    """
    Feature Pyramid Network with MobileNet-DSC (Depthwise Separable Convolution) backbone.
    This lightweight architecture extracts multi-scale feature maps from highly degraded
    inputs, operating efficiently on edge hardware (<8GB VRAM).
    """
    def __init__(self, out_channels=256):
        super(FPNMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2 backbone (lightweight DSC architecture)
        mobilenet = mobilenet_v2(pretrained=True).features
        
        # MobileNetV2 features have 19 layers. We extract features at specific scales:
        # Layer 3: Output shape (24, H/4, W/4)
        # Layer 6: Output shape (32, H/8, W/8)
        # Layer 13: Output shape (96, H/16, W/16)
        # Layer 18: Output shape (1280, H/32, W/32)
        
        self.stage1 = mobilenet[0:4]   # output channels: 24
        self.stage2 = mobilenet[4:7]   # output channels: 32
        self.stage3 = mobilenet[7:14]  # output channels: 96
        self.stage4 = mobilenet[14:19] # output channels: 1280
        
        # Define the FPN using torchvision's utility
        # In_channels list must match the output channels of the selected stages
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[24, 32, 96, 1280],
            out_channels=out_channels
        )
        
    def forward(self, x):
        """
        :param x: Input image tensor of shape (B, C, H, W)
        :return: Dictionary of multi-scale features
        """
        # Extract features from backbone
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        
        # Format for FPN
        features = OrderedDict([
            ('feat1', c1),
            ('feat2', c2),
            ('feat3', c3),
            ('feat4', c4)
        ])
        
        # Pass through FPN neck
        fpn_features = self.fpn(features)
        
        return fpn_features

# Quick test if run as main
if __name__ == "__main__":
    model = FPNMobileNet(out_channels=128)
    # Dummy input representing a batch of 2 degraded images (24x24 upscaled or raw size)
    # The FPN expects sufficient spatial dimensions, so we assume the 24x24 input 
    # is first upsampled to at least 128x128 or 256x256 before FPN processing,
    # or the FPN is designed for small inputs. Let's test with 256x256.
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    
    print("FPN Outputs:")
    for k, v in out.items():
        print(f"{k}: {v.shape}")
