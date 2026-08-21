import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np

# We use InsightFace to extract ArcFace embeddings for the Cosine Similarity metric.
# Note: In a production environment, you might need to build insightface from source or use onnx.
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("WARNING: InsightFace not installed properly. ArcFace Cosine Similarity will be skipped.")

class Evaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        self.face_app = None
        if INSIGHTFACE_AVAILABLE:
            # Initialize ArcFace model via InsightFace for 512-d embeddings
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(256, 256))

    def compute_metrics(self, reconstructed, hr_target):
        """
        Computes PSNR, SSIM, and ArcFace Cosine Similarity.
        
        :param reconstructed: Tensor of shape (B, 3, 256, 256), range [0, 1]
        :param hr_target: Tensor of shape (B, 3, 256, 256), range [0, 1]
        :return: dict of metric scores
        """
        # 1. PSNR & SSIM
        psnr_val = self.psnr_metric(reconstructed, hr_target).item()
        ssim_val = self.ssim_metric(reconstructed, hr_target).item()
        
        cosine_sim = 0.0
        
        # 2. ArcFace Cosine Similarity (Identity Loss)
        if self.face_app is not None:
            batch_size = reconstructed.size(0)
            sim_scores = []
            
            for i in range(batch_size):
                # Convert tensors [0, 1] to numpy BGR [0, 255] for InsightFace
                rec_np = (reconstructed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rec_bgr = rec_np[:, :, ::-1] # RGB to BGR
                
                hr_np = (hr_target[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                hr_bgr = hr_np[:, :, ::-1] # RGB to BGR
                
                # Extract embeddings
                rec_faces = self.face_app.get(rec_bgr)
                hr_faces = self.face_app.get(hr_bgr)
                
                if len(rec_faces) > 0 and len(hr_faces) > 0:
                    rec_emb = torch.tensor(rec_faces[0].embedding)
                    hr_emb = torch.tensor(hr_faces[0].embedding)
                    
                    # Compute Cosine Similarity
                    sim = F.cosine_similarity(rec_emb.unsqueeze(0), hr_emb.unsqueeze(0)).item()
                    sim_scores.append(sim)
                    
            if len(sim_scores) > 0:
                cosine_sim = sum(sim_scores) / len(sim_scores)
                
        return {
            "PSNR": round(psnr_val, 2),
            "SSIM": round(ssim_val, 4),
            "ArcFace_Sim": round(cosine_sim, 4)
        }
