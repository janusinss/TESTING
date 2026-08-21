import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

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

app = FastAPI(title="Forensic DGP Web UI")

# Mount static files (CSS, JS)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model globally (loads once on startup)
print("Loading Forensic DGP Synthesizer...")
model = DGPSynthesizer()
model.eval()
print("Model loaded.")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main UI HTML file."""
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/reconstruct")
async def reconstruct_image(file: UploadFile = File(...)):
    """
    Receives an uploaded CCTV image crop, runs it through the DGP synthesizer,
    and returns the Top-K reconstructed images as base64 strings.
    """
    # 1. Read uploaded image bytes
    contents = await file.read()
    
    # 2. Convert to numpy array via OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Invalid image format."}
        
    # 3. Preprocess to the required degraded tensor format (24x24)
    # Even if the uploaded crop is larger, we simulate the standard sub-32x32 CCTV bounding box
    img_bgr = cv2.resize(img_bgr, (24, 24))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0) # Add batch dimension -> (1, 3, 24, 24)
    
    # 4. Generate Top-K reconstructions
    k = 3
    # Note: In production, we don't have hr_target. We pass the upscaled input as a dummy target,
    # or rely on a no-reference aesthetic metric. For this pipeline demo, we pass the upscaled 
    # degraded image as the target to keep the Morphological Loss functional (it will find the 
    # approximate face shape of the degraded image).
    dummy_target = torch.nn.functional.interpolate(input_tensor, size=(256, 256), mode='bilinear')
    
    top_k_reconstructions, scores = model.generate_top_k(input_tensor, dummy_target, k=k)
    
    # 5. Convert output tensors to base64 strings for the frontend
    result_images = []
    for idx, rec_tensor in enumerate(top_k_reconstructions):
        # tensor is (1, 3, 256, 256) in [0, 1] range
        rec_img_np = (rec_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        
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
