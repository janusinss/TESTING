import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import face_alignment

from degradation import (
    apply_optical_motion_blur,
    apply_atmospheric_scattering,
    apply_spatial_downsampling,
    apply_thermal_noise,
    apply_h264_quantization
)

class DegradedFacesDataset(Dataset):
    """
    PyTorch Dataset that dynamically generates heavily degraded low-resolution
    CCTV-like faces paired with high-resolution pristine targets and 68-point
    facial landmarks.
    """
    def __init__(self, root_dir, transform=None, target_size=(24, 24), hr_size=(256, 256)):
        """
        :param root_dir: Path to directory containing high-res face images (e.g., FFHQ).
        :param transform: Optional torchvision transforms.
        :param target_size: The degraded sub-32x32 target size.
        :param hr_size: The standardized high-res size for the pristine image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.hr_size = hr_size
        
        # Load all valid image file paths
        valid_exts = {'.png', '.jpg', '.jpeg'}
        self.image_paths = []
        if os.path.exists(root_dir):
            for fname in os.listdir(root_dir):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    self.image_paths.append(os.path.join(root_dir, fname))
                    
        # Initialize face-alignment network (FAN)
        # Using CPU by default for the dataloader to avoid GPU memory conflicts, 
        # but can be switched to 'cuda' if sufficient VRAM is available.
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

    def __len__(self):
        return len(self.image_paths)
        
    def _get_landmarks(self, image):
        """
        Extracts 68-point Cartesian coordinates.
        :param image: numpy array of the image (RGB format expected for face-alignment)
        :return: numpy array of shape (68, 2)
        """
        # face_alignment expects RGB images (values 0-255)
        preds = self.fa.get_landmarks(image)
        
        if preds is None or len(preds) == 0:
            return np.zeros((68, 2), dtype=np.float32)
            
        # Get landmarks for the first face found
        coords = preds[0].astype(np.float32)
        return coords

    def apply_compound_degradation(self, image):
        """
        Applies the four-stage compound degradation model from Chapter 3.
        I_LR = C_H264( ((I_HR * K_M) down_s + N(mu, sigma^2)), Q_p )
        """
        # 1. Optical Motion Blur (K_M)
        # Randomize displacement and angle for dynamic generation
        d = np.random.randint(2, 10)
        theta = np.random.uniform(0, 360)
        img_deg = apply_optical_motion_blur(image, d, theta)
        
        # 2. Atmospheric Scattering Simulation
        t = np.random.uniform(0.3, 0.8) # 0.3 is heavy fog, 0.8 is light
        A = np.random.uniform(0.6, 1.0)
        img_deg = apply_atmospheric_scattering(img_deg, t, A)
        
        # Convert back to uint8 for cv2 processing if needed
        if img_deg.dtype != np.uint8:
            img_deg = (np.clip(img_deg, 0.0, 1.0) * 255).astype(np.uint8)
            
        # 3. Spatial Downsampling (down_s)
        img_deg = apply_spatial_downsampling(img_deg, self.target_size)
        
        # 4. Sensor Thermal Noise Injection (N(0, sigma^2))
        std = np.random.uniform(5, 25)
        img_deg = apply_thermal_noise(img_deg, mean=0, std=std)
        
        # 5. Aggressive H.264 Quantization (Q_p >= 35)
        qp = np.random.randint(35, 51)
        img_deg = apply_h264_quantization(img_deg, qp=qp)
        
        return img_deg

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load high-res pristine image I_HR
        i_hr_bgr = cv2.imread(img_path)
        if i_hr_bgr is None:
            raise ValueError(f"Failed to load image at {img_path}")
            
        # Standardize size
        i_hr_bgr = cv2.resize(i_hr_bgr, self.hr_size)
        
        # Convert I_HR to RGB (since cv2 loads in BGR) for landmarks and output
        i_hr_rgb = cv2.cvtColor(i_hr_bgr, cv2.COLOR_BGR2RGB)
        
        # Get 68-point landmarks from I_HR
        landmarks = self._get_landmarks(i_hr_rgb)
        
        # Generate I_LR (Degraded) from BGR (for opencv operations)
        i_lr_bgr = self.apply_compound_degradation(i_hr_bgr)
        i_lr_rgb = cv2.cvtColor(i_lr_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to torch Tensors (C, H, W format)
        i_hr_tensor = torch.from_numpy(i_hr_rgb).permute(2, 0, 1).float() / 255.0
        i_lr_tensor = torch.from_numpy(i_lr_rgb).permute(2, 0, 1).float() / 255.0
        landmarks_tensor = torch.from_numpy(landmarks)
        
        if self.transform:
            i_hr_tensor = self.transform(i_hr_tensor)
            i_lr_tensor = self.transform(i_lr_tensor)
            
        return i_lr_tensor, i_hr_tensor, landmarks_tensor
