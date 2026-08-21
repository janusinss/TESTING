import torch
import torch.nn as nn
import face_alignment

class MorphologicalFANLoss(nn.Module):
    """
    Morphological 68-Point FAN Loss.
    Uses the pre-trained Face Alignment Network (FAN) to extract spatial landmarks
    and compute the loss between the generated image and the ground truth.
    This restricts the generator from hallucinating facial geometries that deviate
    from the subject's true morphology.
    """
    def __init__(self, device='cpu'):
        super(MorphologicalFANLoss, self).__init__()
        self.device = device
        
        # Initialize the FAN model (returns 2D heatmaps)
        # We load it but avoid computing gradients for the FAN network itself
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        self.fan_net = fa.face_alignment_net
        self.fan_net.eval()
        
        # Freeze FAN network weights
        for param in self.fan_net.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()

    def forward(self, generated_img, target_img):
        """
        Computes the morphological loss.
        :param generated_img: Reconstructed high-res tensor (B, 3, 256, 256), values [0, 1] or [-1, 1]
        :param target_img: Ground truth pristine high-res tensor (B, 3, 256, 256)
        :return: Scalar loss value
        """
        # Ensure images are properly sized for FAN (FAN typically expects 256x256)
        # FAN expects inputs normalized in a specific way, typically around 0-255 or -1 to 1. 
        # For torch modules, it's usually standard image ranges. 
        # We'll just pass them through the network to get the heatmaps.
        
        # We must keep this differentiable for the generated_img!
        # generated_img requires_grad=True
        
        # Extract heatmaps
        gen_heatmaps = self.fan_net(generated_img)
        
        # Target heatmaps don't need gradients
        with torch.no_grad():
            target_heatmaps = self.fan_net(target_img)
            
        # The output of FAN is usually a list of tensors or a single tensor of heatmaps
        # We can compute the Mean Squared Error between the heatmaps.
        # This is a differentiable proxy for the 68-point euclidean distance.
        if isinstance(gen_heatmaps, list) or isinstance(gen_heatmaps, tuple):
            # If it returns multiple scales (like in some hourglass networks), use the final one
            loss = self.mse_loss(gen_heatmaps[-1], target_heatmaps[-1])
        else:
            loss = self.mse_loss(gen_heatmaps, target_heatmaps)
            
        return loss

# Quick test if run as main
if __name__ == "__main__":
    import os
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    
    loss_fn = MorphologicalFANLoss()
    # Dummy images
    gen_img = torch.randn(1, 3, 256, 256, requires_grad=True)
    tar_img = torch.randn(1, 3, 256, 256)
    
    loss = loss_fn(gen_img, tar_img)
    print(f"Computed Morphological Loss: {loss.item()}")
    
    # Check if gradients flow back to the generated image
    loss.backward()
    print(f"Gradient flowing to generated image: {gen_img.grad is not None}")
