import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

import io
import cv2
import base64
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from models import DGPSynthesizer
from degradation import adaptive_cctv_denoise

app = FastAPI(title="Optimal Face Restoration Web UI")

# Mount static files (CSS, JS)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model globally (loads once on startup)
print("Loading Optimal Generative Face Restoration Synthesizer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGPSynthesizer().to(device)

# Auto-detect best model weights
checkpoint_path = "mapped_deblurgan.pth"
if os.path.exists("checkpoints/dgp_epoch_1.pth"):
    checkpoint_path = "checkpoints/dgp_epoch_1.pth"
elif os.path.exists("checkpoints/dgp_transfer_epoch_5.pth"):
    checkpoint_path = "checkpoints/dgp_transfer_epoch_5.pth"

if os.path.exists(checkpoint_path):
    print(f"SUCCESS: Loading pre-trained intelligence from {checkpoint_path}...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
        print("Model loaded with 100% strict matching.")
    except Exception as e:
        print(f"Notice: {e}. Falling back to strict=False...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
else:
    print(f"WARNING: {checkpoint_path} not found! Please place weights file in root or checkpoints/.")

model.eval()
print("Optimal Face Restoration Engine ready.")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main UI HTML file."""
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/reconstruct")
async def reconstruct_image(file: UploadFile = File(...)):
    """
    Receives an uploaded CCTV image crop, applies adaptive edge-preserving pre-denoising,
    synthesizes high-resolution reconstructions via DeblurGAN-v2 multi-scale nearest fusion,
    and returns razor-sharp Top-K candidates.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Invalid image format."}
        
    # 1. Adaptive Edge-Preserving Pre-Denoising (Bilateral / Median)
    # Eliminates high-frequency thermal sensor noise before super-resolution
    img_clean_bgr = adaptive_cctv_denoise(img_bgr)
    
    # 2. Preprocess to standard sub-32x32 CCTV bounding box (24x24)
    img_low_bgr = cv2.resize(img_clean_bgr, (24, 24))
    img_rgb = cv2.cvtColor(img_low_bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] tensor
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device) # (1, 3, 24, 24)
    
    # 3. Generate Top-K reconstructions
    k = 3
    dummy_target = torch.nn.functional.interpolate(input_tensor, size=(256, 256), mode='bilinear').to(device)
    top_k_reconstructions, scores = model.generate_top_k(input_tensor, dummy_target, k=k)
    
    # 4. Convert output tensors to base64 strings with guided detail enhancement
    result_images = []
    for idx, rec_tensor in enumerate(top_k_reconstructions):
        # tensor is (1, 3, 256, 256) in [0, 1] range
        rec_img_np = (rec_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        
        # Subtle unsharp masking to enhance eye, iris, and facial edge definition
        blurred = cv2.GaussianBlur(rec_img_np, (0, 0), 1.5)
        rec_img_np = cv2.addWeighted(rec_img_np, 1.3, blurred, -0.3, 0)
        rec_img_np = np.clip(rec_img_np, 0, 255).astype(np.uint8)
        
        # Encode back to PNG buffer
        success, encoded_img = cv2.imencode('.png', cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2BGR))
        if success:
            base64_str = base64.b64encode(encoded_img).decode('utf-8')
            result_images.append({
                "rank": idx + 1,
                "score": round(scores[idx], 4),
                "image_data": f"data:image/png;base64,{base64_str}"
            })
            
    return {"results": result_images}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
