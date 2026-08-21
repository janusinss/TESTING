import cv2
import numpy as np
import torch
import math

def apply_optical_motion_blur(image, d, theta):
    """
    Simulates optical motion blur.
    :param image: Input image (numpy array, HxWxC or HxW)
    :param d: Displacement distance (pixels)
    :param theta: Trajectory angle (degrees)
    :return: Motion-blurred image
    """
    # Ensure d is an integer and > 0
    d = max(1, int(d))
    
    # Create the motion blur kernel
    kernel = np.zeros((d, d), dtype=np.float32)
    center = d // 2
    
    # Calculate the end point of the line based on the angle
    theta_rad = math.radians(theta)
    x = int(center * math.cos(theta_rad))
    y = int(center * math.sin(theta_rad))
    
    # Draw a line on the kernel to represent the motion path
    cv2.line(kernel, (center - x, center - y), (center + x, center + y), 1.0, 1)
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply the 2D filter (convolution)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def apply_atmospheric_scattering(image, t, A):
    """
    Simulates tropical rainfall, lens fogging, and humidity using the transmission model:
    I(x) = J(x)t(x) + A(1 - t(x))
    :param image: Input pristine image J(x) (numpy array, float32, range [0, 1])
    :param t: Transmission map (float or numpy array same size as image, range [0, 1]).
              Lower t means more scattering/fog.
    :param A: Atmospheric light (float or numpy array, typically between 0.5 and 1.0)
    :return: Scattered image I(x)
    """
    # Ensure image is float in [0, 1] for the math to work correctly
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        
    degraded = image * t + A * (1 - t)
    
    # Clip and convert back if necessary
    degraded = np.clip(degraded, 0.0, 1.0)
    return degraded

def apply_spatial_downsampling(image, target_size=(24, 24)):
    """
    Bicubic interpolation reducing ground truth to sub-32x32 resolution.
    :param image: Input image
    :param target_size: Tuple (width, height) for the downsampled size
    :return: Downsampled image
    """
    downsampled = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return downsampled

def apply_thermal_noise(image, mean=0, std=10):
    """
    Injects zero-mean additive Gaussian noise N(0, sigma^2).
    :param image: Input image (numpy array, uint8 or float32)
    :param mean: Mean of the Gaussian noise
    :param std: Standard deviation of the Gaussian noise
    :return: Noisy image
    """
    is_uint8 = image.dtype == np.uint8
    if is_uint8:
        image = image.astype(np.float32)
        
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image + noise
    
    if is_uint8:
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    else:
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
    return noisy_image

def apply_h264_quantization(image, qp=35):
    """
    Approximates aggressive H.264 quantization (Q_p >= 35).
    Since true H.264 encoding requires a video codec, we simulate it here
    by applying heavy JPEG compression, which uses a similar DCT block-based 
    quantization and chroma subsampling approach.
    :param image: Input image (numpy array, uint8)
    :param qp: Quantization parameter approximation. Higher QP = lower quality.
               (Typically H.264 QP ranges 0-51. We map it roughly to JPEG quality.)
    :return: Quantized image
    """
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        
    # Map QP (approx 0-51) to JPEG quality (1-100). 
    # QP 35 is very lossy. A simple inverse linear mapping:
    # QP 0 -> Quality 100
    # QP 51 -> Quality 10
    quality = max(1, int(100 - (qp / 51.0) * 90))
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    if result:
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    return image
