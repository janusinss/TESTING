import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from dataloader import get_dataloader
from models import DGPSynthesizer, MorphologicalFANLoss
from evaluation import Evaluator

def train(args):
    # Setup Device (Works on Cloud GPU or Local CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Initialize DataLoader
    print("Initializing DataLoader...")
    dataloader = get_dataloader(root_dir=args.data_dir, batch_size=args.batch_size, shuffle=True)
    
    # 2. Initialize Models
    print("Initializing Models...")
    model = DGPSynthesizer().to(device)
    fan_loss_fn = MorphologicalFANLoss(device=str(device))
    l1_loss_fn = nn.L1Loss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Evaluator
    evaluator = Evaluator(device=device)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # 3. Training Loop
    print(f"Starting Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (low_res, hr_target, landmarks) in enumerate(progress_bar):
            low_res = low_res.to(device)
            hr_target = hr_target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(low_res)
            
            # Compute Losses
            # Photometric Loss (L1) ensures general pixel alignment
            l1_loss = l1_loss_fn(reconstructed, hr_target)
            
            # Morphological FAN Loss ensures geometric/biometric alignment (The core thesis contribution)
            fan_loss = fan_loss_fn(reconstructed, hr_target)
            
            # Total Loss = Photometric + Lambda * Morphological
            total_loss = l1_loss + (args.lambda_fan * fan_loss)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
            
            # For Dry Run, we break after 1 batch
            if args.dry_run:
                print("\n[Dry Run] Successfully completed 1 batch. Breaking loop.")
                break
                
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
        
        # 4. Evaluation Phase (End of Epoch)
        model.eval()
        with torch.no_grad():
            metrics = evaluator.compute_metrics(reconstructed, hr_target)
            print(f"Validation Metrics -> PSNR: {metrics['PSNR']} | SSIM: {metrics['SSIM']} | ArcFace: {metrics['ArcFace_Sim']}")
            
        # Save Checkpoint
        checkpoint_path = f"checkpoints/dgp_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}\n")
        
        if args.dry_run:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Forensic Deep Generative Prior Model")
    parser.add_argument("--data_dir", type=str, default="dataset/ffhq", help="Path to FFHQ dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (Keep small for <8GB VRAM)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_fan", type=float, default=0.1, help="Weight for Morphological FAN Loss")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 batch to verify the pipeline")
    
    args = parser.parse_args()
    train(args)
