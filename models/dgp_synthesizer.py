import torch
import torch.nn as nn
import torch.nn.functional as F
from .fpn_mobilenet import FPNMobileNet
from .morphological_loss import MorphologicalFANLoss

class DGPSynthesizer(nn.Module):
    """
    Forensic Deep Generative Prior (DGP) Synthesizer.
    Utilizes the FPN-MobileNet backbone (inspired by DeblurGAN-v2) to extract
    multi-scale features, fuses them, and reconstructs the pristine high-res face.
    """
    def __init__(self, fpn_out_channels=128):
        super(DGPSynthesizer, self).__init__()
        
        # Multi-scale feature extractor
        self.fpn = FPNMobileNet(out_channels=fpn_out_channels)
        
        # Feature fusion and upsampling layers
        # The FPN outputs features at 4 scales. If input is 256x256:
        # feat1: 64x64, feat2: 32x32, feat3: 16x16, feat4: 8x8
        # We will upsample and sum them all to 64x64, then do two more upsample blocks to reach 256x256
        
        self.smooth1 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        
        # Upsampling blocks to go from 64x64 -> 128x128 -> 256x256
        self.upblock1 = nn.Sequential(
            nn.ConvTranspose2d(fpn_out_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upblock2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer mapping back to 3 channels (RGB)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh() # Output in range [-1, 1]
        )

    def freeze_backbone(self):
        """Freezes the FPN MobileNet backbone to preserve transfer-learned priors."""
        for stage in [self.fpn.stage1, self.fpn.stage2, self.fpn.stage3, self.fpn.stage4]:
            for param in stage.parameters():
                param.requires_grad = False
        print("Backbone (MobileNetV2) successfully frozen for Transfer Learning.")

    def forward(self, x):
        """
        :param x: Degraded input tensor (B, 3, 24, 24)
        :return: Reconstructed pristine tensor (B, 3, 256, 256)
        """
        # Upsample the tiny 24x24 degraded input to 256x256 before processing
        # This provides the spatial canvas for the FPN to work on
        x_up = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Extract FPN features
        features = self.fpn(x_up)
        
        # Smooth the FPN outputs
        f1 = self.smooth1(features['feat1']) # 64x64
        f2 = self.smooth2(features['feat2']) # 32x32
        f3 = self.smooth3(features['feat3']) # 16x16
        f4 = self.smooth4(features['feat4']) # 8x8
        
        # Fuse them by progressively upsampling and adding
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f3_fused = f3 + f4_up
        
        f3_up = F.interpolate(f3_fused, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f2_fused = f2 + f3_up
        
        f2_up = F.interpolate(f2_fused, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f1_fused = f1 + f2_up # Size: 64x64
        
        # Final upsampling to 256x256
        out = self.upblock1(f1_fused) # 128x128
        out = self.upblock2(out)      # 256x256
        out = self.final_conv(out)    # 3 channels
        
        # Scale back to [0, 1] for image formats, assuming input x was [0, 1]
        out = (out + 1.0) / 2.0
        
        return out

    def generate_top_k(self, degraded_img, hr_target, k=5, noise_std=0.05):
        """
        Implements Top-K Sampling.
        Instead of returning a single hallucination, we run the model K times 
        with slight latent perturbations. We then score each reconstruction 
        using the Morphological FAN Loss and return the top K results.
        
        :param degraded_img: Tensor of shape (1, 3, 24, 24)
        :param hr_target: Ground truth tensor (1, 3, 256, 256) to compare against for scoring
        :param k: Number of top samples to return
        """
        # In a real deployed edge scenario, hr_target wouldn't be available, 
        # and scoring would use a no-reference metric or structural prior. 
        # For training/validation, we use the hr_target with Morphological Loss.
        
        loss_fn = MorphologicalFANLoss(device=str(degraded_img.device))
        
        reconstructions = []
        scores = []
        
        # Upsample base input
        x_up = F.interpolate(degraded_img, size=(256, 256), mode='bilinear', align_corners=False)
        
        with torch.no_grad(): # Inference mode
            for i in range(max(10, k)): # Generate 10 variations, pick top K
                # Add slight perturbation to the input to explore the generative prior space
                noise = torch.randn_like(x_up) * noise_std
                noisy_input = torch.clamp(x_up + noise, 0.0, 1.0)
                
                # Generate
                features = self.fpn(noisy_input)
                # ... (inline forward pass for perturbed features)
                f1, f2, f3, f4 = self.smooth1(features['feat1']), self.smooth2(features['feat2']), self.smooth3(features['feat3']), self.smooth4(features['feat4'])
                f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
                f3_up = F.interpolate(f3 + f4_up, size=f2.shape[2:], mode='bilinear', align_corners=False)
                f2_up = F.interpolate(f2 + f3_up, size=f1.shape[2:], mode='bilinear', align_corners=False)
                f1_fused = f1 + f2_up
                out = self.final_conv(self.upblock2(self.upblock1(f1_fused)))
                out = (out + 1.0) / 2.0
                
                # Score using Morphological Loss
                score = loss_fn(out, hr_target).item()
                
                reconstructions.append(out)
                scores.append(score)
                
        # Sort by best (lowest) score
        scored_reconstructions = list(zip(scores, reconstructions))
        scored_reconstructions.sort(key=lambda x: x[0])
        
        # Return top K
        top_k_reconstructions = [x[1] for x in scored_reconstructions[:k]]
        top_k_scores = [x[0] for x in scored_reconstructions[:k]]
        
        return top_k_reconstructions, top_k_scores
