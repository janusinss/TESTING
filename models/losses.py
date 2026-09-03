import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

from torchvision.models import vgg19, VGG19_Weights
from .morphological_loss import MorphologicalFANLoss

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Smooth L1).
    Outperforms standard L1/L2 by remaining robust to edge outliers while
    providing continuous, differentiable gradients near zero.
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps2))
        return loss

class ColorLoss(nn.Module):
    """
    Color Consistency Loss.
    Penalizes chromatic deviations on low-frequency average-pooled feature maps.
    Effectively eliminates color collapse and chromatic drift (green/yellow/blue tints).
    """
    def __init__(self, kernel_size=11):
        super(ColorLoss, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x, y):
        return F.l1_loss(self.pool(x), self.pool(y))

class VGGPerceptualLoss(nn.Module):
    """
    VGG19 Deep Perceptual Loss (Layer conv3_3).
    Measures feature representations in deep semantic space rather than raw pixels.
    Forces the network to generate sharp hair, distinct iris contours, and realistic skin texture.
    """
    def __init__(self, device='cpu'):
        super(VGGPerceptualLoss, self).__init__()
        self.device = device
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, x, y):
        # Scale to ImageNet distribution
        x_norm = (x - self.mean) / self.std
        y_norm = (y - self.mean) / self.std
        feat_x = self.vgg(x_norm)
        with torch.no_grad():
            feat_y = self.vgg(y_norm)
        return F.l1_loss(feat_x, feat_y)

class OptimalFaceRestorationLoss(nn.Module):
    """
    Composite Multi-Component Loss for Optimal Face Reconstruction:
    L_total = L_charb + lambda_vgg * L_vgg + lambda_color * L_color + lambda_fan * L_fan
    """
    def __init__(self, device='cpu', lambda_vgg=0.15, lambda_color=0.05, lambda_fan=0.05):
        super(OptimalFaceRestorationLoss, self).__init__()
        self.charb_loss = CharbonnierLoss()
        self.vgg_loss = VGGPerceptualLoss(device=device)
        self.color_loss = ColorLoss()
        self.fan_loss = MorphologicalFANLoss(device=str(device))
        
        self.lambda_vgg = lambda_vgg
        self.lambda_color = lambda_color
        self.lambda_fan = lambda_fan

    def forward(self, reconstructed, hr_target):
        l_charb = self.charb_loss(reconstructed, hr_target)
        l_vgg = self.vgg_loss(reconstructed, hr_target)
        l_color = self.color_loss(reconstructed, hr_target)
        l_fan = self.fan_loss(reconstructed, hr_target)

        total_loss = l_charb + (self.lambda_vgg * l_vgg) + (self.lambda_color * l_color) + (self.lambda_fan * l_fan)
        return total_loss, {
            "Charbonnier": l_charb.item(),
            "VGG": l_vgg.item(),
            "Color": l_color.item(),
            "FAN": l_fan.item(),
            "Total": total_loss.item()
        }
