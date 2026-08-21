import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataloader import get_dataloader

def test_pipeline():
    # Make sure we have a sample directory
    sample_dir = "dataset/ffhq"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Check if there are images, if not, print a message and exit
    valid_exts = {'.png', '.jpg', '.jpeg'}
    images_exist = any(os.path.splitext(f)[1].lower() in valid_exts for f in os.listdir(sample_dir))
    
    if not images_exist:
        print(f"Please place at least one high-res face image in '{sample_dir}' to run this test.")
        return
        
    print(f"Loading data from {sample_dir}...")
    dataloader = get_dataloader(sample_dir, batch_size=1, num_workers=0) # num_workers 0 for simple test
    
    # Get one batch
    for i_lr, i_hr, landmarks in dataloader:
        print(f"High-res tensor shape: {i_hr.shape}")
        print(f"Low-res tensor shape: {i_lr.shape}")
        print(f"Landmarks tensor shape: {landmarks.shape}")
        
        # Convert first item in batch back to numpy for visualization, make contiguous with .copy()
        hr_img = (i_hr[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        lr_img = (i_lr[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        lm_coords = landmarks[0].numpy()
        
        # Draw landmarks on HR image
        for (x, y) in lm_coords:
            if x > 0 and y > 0: # Only draw if landmark exists
                cv2.circle(hr_img, (int(x), int(y)), 2, (0, 255, 0), -1)
                
        # Resize LR back up to HR size for side-by-side comparison (just to see the pixels clearly)
        lr_upscaled = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(hr_img)
        axes[0].set_title("Original High-Res + Landmarks")
        axes[0].axis('off')
        
        axes[1].imshow(lr_upscaled)
        axes[1].set_title("Degraded (sub-32x32 -> Nearest Upscale)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("test_output_visualization.png")
        print("Test complete. Saved visualization to 'test_output_visualization.png'.")
        break # Only process one batch for the test

if __name__ == "__main__":
    test_pipeline()
