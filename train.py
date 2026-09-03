import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from dataloader import get_dataloader
from models import DGPSynthesizer, OptimalFaceRestorationLoss
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
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint/transfer weights: {args.resume_from}")
        try:
            model.load_state_dict(torch.load(args.resume_from, map_location=device), strict=True)
            print("SUCCESS: 100% of pre-trained weights loaded with strict=True!")
        except Exception as e:
            print(f"Notice on strict load: {e}. Falling back to strict=False...")
            model.load_state_dict(torch.load(args.resume_from, map_location=device), strict=False)
        
    # Optimizer configuration
    if args.transfer_learning:
        # Differential Learning Rate: backbone fine-tunes gently, head learns active synthesis
        backbone_params = list(model.fpn.features.parameters())
        head_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr}
        ], betas=(0.9, 0.999))
        print(f"Applied Differential Learning Rate (Backbone: {args.lr * 0.1:.1e}, Head: {args.lr:.1e})")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        
    # Composite Optimal Loss Function
    criterion = OptimalFaceRestorationLoss(
        device=device,
        lambda_vgg=args.lambda_vgg,
        lambda_color=args.lambda_color,
        lambda_fan=args.lambda_fan
    )
    
    # Evaluator
    evaluator = Evaluator(device=device)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 3. Training Loop
    print(f"Starting Training from Epoch {args.start_epoch} to {args.epochs}...")
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (low_res, hr_target, landmarks) in enumerate(progress_bar):
            low_res = low_res.to(device)
            hr_target = hr_target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(low_res)
            
            # Compute Composite Loss (Charbonnier + VGG19 + Color + FAN)
            total_loss, loss_dict = criterion(reconstructed, hr_target)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "VGG": f"{loss_dict['VGG']:.3f}",
                "Col": f"{loss_dict['Color']:.3f}"
            })
            
            # For Dry Run, break after 1 batch
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
    parser = argparse.ArgumentParser(description="Train the Optimal Deep Generative Prior Face Restoration Model")
    parser.add_argument("--data_dir", type=str, default="dataset/ffhq", help="Path to FFHQ dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (Keep small for <8GB VRAM)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--start_epoch", type=int, default=1, help="Epoch to start counting from")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pth file to resume from")
    parser.add_argument("--transfer_learning", action="store_true", help="Enable differential learning rate for transfer learning")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for generative head")
    parser.add_argument("--lambda_vgg", type=float, default=0.15, help="Weight for VGG19 Perceptual Loss")
    parser.add_argument("--lambda_color", type=float, default=0.05, help="Weight for Color Consistency Loss")
    parser.add_argument("--lambda_fan", type=float, default=0.05, help="Weight for Morphological FAN Loss")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 batch to verify the pipeline")
    
    args = parser.parse_args()
    train(args)
