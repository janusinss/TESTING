# Forensic Deep Generative Prior (DGP) for CCTV Video

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

> **Thesis Title**: Forensic Deep Generative Prior Face Reconstruction for Degraded CCTV Video in Zamboanga City  
> **Institution**: Western Mindanao State University, College of Computing Studies  

This repository contains the software implementation for a **Morphologically Constrained Deep Generative Prior (DGP)** designed for forensic digital investigation. It aims to reconstruct highly degraded, sub-32x32 pixel CCTV facial crops (commonly afflicted by severe H.264 macroblocking and environmental noise) into morphologically accurate high-resolution representations without triggering AI hallucination or biometric shift.

## Features

1. **Degradation Data Engine**: A highly specialized pipeline simulating the exact tropical scattering, sensor noise, and aggressive H.264 bitrate starvation experienced by municipal CCTV grids.
2. **Morphologically Constrained DGP Synthesizer**: 
   - A lightweight `MobileNetV2` Feature Pyramid Network (FPN).
   - Constrained by a 68-point biometric **Morphological FAN Loss** to ensure the generated face mathematically adheres to the original Euclidean geometry of the suspect.
   - **Top-K Probabilistic Sampling** to provide investigators with a statistical spread of possible facial structures instead of forcing a single deterministic guess.
3. **Forensic Evaluation Metrics**: Built-in scripts to benchmark Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), and **ArcFace Cosine Similarity** to legally validate identity preservation.
4. **Air-Gapped Forensic Web Interface**: A local FastAPI web application built with a brutalist/utilitarian UI to serve the models for law enforcement integration.

---

##  Getting Started (Local Deployment)

### Prerequisites
- Python 3.10+
- (Optional) CUDA-enabled NVIDIA GPU for faster training/inference

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forensic-dgp.git
   cd forensic-dgp
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Interface
To serve the forensic web application locally:
```bash
uvicorn app:app --reload
```
Navigate to `http://localhost:8000` in your web browser.

---

##  Cloud Deployment (Google Cloud / Docker)

For training the model on the full FFHQ dataset using Google Cloud Compute Engine (Deep Learning VMs) with GPU acceleration, a `Dockerfile` is provided.

### Build the Docker Image
```bash
docker build -t forensic-dgp .
```

### Run the Docker Container (Training)
Assuming you have mounted your dataset volume to `/app/dataset`:
```bash
docker run --gpus all -v /path/to/dataset:/app/dataset forensic-dgp python train.py --epochs 50 --batch_size 16
```

---

##  Training and Evaluation

### Dataset Setup
Download the [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset) and place the high-resolution images in `dataset/ffhq/`.

### Run the Training Loop
```bash
python train.py --epochs 10 --batch_size 8 --lambda_fan 0.1
```
During training, the model dynamically degrades the FFHQ images to sub-32x32 arrays and computes both Photometric (L1) and Morphological (FAN) losses.

---

##  Legal Context
This software was developed specifically to address the digital forensic admissibility constraints outlined by the *Philippine Rules on Electronic Evidence*. By strictly penalizing morphological deviation, the system reduces the risk of generating synthetic data fabrications ("hallucinations"), thereby maintaining the digital chain-of-custody for investigative operations.
