# Use the official PyTorch image with CUDA support (ideal for Google Cloud GPU VMs)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies (required for OpenCV and InsightFace)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
# (The .dockerignore will prevent venv, dataset, and checkpoints from copying)
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# By default, start the FastAPI Web UI
# You can override this command when running the container to run train.py instead
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
