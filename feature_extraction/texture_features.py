"""
Texture Feature Extraction Module

This module implements texture-based descriptors:
1. Gabor Filters (multiple orientations and scales)
2. Tamura Features (coarseness, contrast, directionality)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from scipy import ndimage


def create_gabor_kernels(ksize: int = 21, 
                        orientations: List[float] = None,
                        scales: List[float] = None) -> List[np.ndarray]:
    """
    Create a bank of Gabor filter kernels.
    
    Args:
        ksize: Kernel size
        orientations: List of orientations in degrees (default: [0, 45, 90, 135])
        scales: List of scales (wavelengths) for the filters
    
    Returns:
        List of Gabor kernels
    """
    if orientations is None:
        orientations = [0, 45, 90, 135]
    
    if scales is None:
        scales = [3, 5, 7]
    
    kernels = []
    for theta in orientations:
        theta_rad = np.deg2rad(theta)
        for lambd in scales:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), 
                sigma=5.0, 
                theta=theta_rad, 
                lambd=lambd, 
                gamma=0.5, 
                psi=0
            )
            kernels.append(kernel)
    
    return kernels


def extract_gabor_features(image: np.ndarray) -> np.ndarray:
    """
    Extract Gabor filter features from an image.
    
    For each Gabor filter, we compute the mean and standard deviation
    of the filter response, creating a texture descriptor.
    
    Args:
        image: Input image (grayscale or color)
    
    Returns:
        Array of Gabor features (mean and std for each filter)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize image
    gray = gray.astype(np.float32) / 255.0
    
    # Create Gabor kernels
    kernels = create_gabor_kernels()
    
    features = []
    for kernel in kernels:
        # Apply Gabor filter
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        
        # Compute statistics
        mean = np.mean(filtered)
        std = np.std(filtered)
        
        features.extend([mean, std])
    
    return np.array(features)


def compute_tamura_coarseness(image: np.ndarray, kmax: int = 5) -> float:
    """
    Compute Tamura coarseness feature.
    
    Coarseness measures the texture granularity. Larger values indicate
    coarser textures.
    
    Args:
        image: Grayscale image
        kmax: Maximum window size parameter (2^kmax x 2^kmax)
    
    Returns:
        Coarseness value
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    # Compute average over neighborhoods of different sizes
    A_k = []
    for k in range(kmax):
        window_size = 2 ** k
        # Use uniform filter for averaging
        avg = ndimage.uniform_filter(gray, size=window_size)
        A_k.append(avg)
    
    # Compute differences between non-overlapping neighborhoods
    E_h = np.zeros((kmax, h, w))
    E_v = np.zeros((kmax, h, w))
    
    for k in range(kmax):
        window_size = 2 ** k
        # Horizontal and vertical differences
        E_h[k] = np.abs(
            np.roll(A_k[k], window_size, axis=1) - 
            np.roll(A_k[k], -window_size, axis=1)
        )
        E_v[k] = np.abs(
            np.roll(A_k[k], window_size, axis=0) - 
            np.roll(A_k[k], -window_size, axis=0)
        )
    
    # Find the k that maximizes E in either direction for each pixel
    E_max = np.maximum(E_h, E_v)
    k_best = np.argmax(E_max, axis=0)
    
    # Compute coarseness
    S_best = 2 ** k_best
    coarseness = np.mean(S_best)
    
    return float(coarseness)


def compute_tamura_contrast(image: np.ndarray) -> float:
    """
    Compute Tamura contrast feature.
    
    Contrast measures the dynamic range of gray levels.
    
    Args:
        image: Grayscale image
    
    Returns:
        Contrast value
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)
    
    # Compute standard deviation and kurtosis
    mu = np.mean(gray)
    sigma = np.std(gray)
    
    # Fourth moment (kurtosis)
    mu4 = np.mean((gray - mu) ** 4)
    
    if sigma == 0:
        return 0.0
    
    # Contrast formula
    alpha4 = mu4 / (sigma ** 4)
    contrast = sigma / (alpha4 ** 0.25) if alpha4 > 0 else 0
    
    return float(contrast)


def compute_tamura_directionality(image: np.ndarray, num_bins: int = 16) -> Tuple[float, np.ndarray]:
    """
    Compute Tamura directionality feature.
    
    Directionality measures the degree to which the texture has a dominant direction.
    
    Args:
        image: Grayscale image
        num_bins: Number of bins for direction histogram
    
    Returns:
        Tuple of (directionality_value, direction_histogram)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)
    
    # Compute gradients using Sobel operators
    delta_h = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    delta_v = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    delta_g = np.sqrt(delta_h ** 2 + delta_v ** 2)
    
    # Compute gradient direction
    theta = np.arctan2(delta_v, delta_h)
    theta = (theta + np.pi) * (180 / np.pi)  # Convert to degrees [0, 360)
    
    # Quantize directions
    theta = theta % 180  # Use 180 degrees (0-180)
    
    # Only consider pixels with significant gradient
    threshold = np.mean(delta_g) * 0.1
    mask = delta_g > threshold
    
    # Build histogram
    if np.sum(mask) > 0:
        hist, bin_edges = np.histogram(
            theta[mask], 
            bins=num_bins, 
            range=(0, 180)
        )
        hist = hist.astype(np.float32)
        hist = hist / np.sum(hist)  # Normalize
    else:
        hist = np.zeros(num_bins, dtype=np.float32)
    
    # Compute directionality measure (peakedness of histogram)
    # Higher values indicate stronger directionality
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks in histogram
    if np.max(hist) > 0:
        # Directionality is inversely related to histogram variance
        mean_angle = np.sum(bin_centers * hist)
        variance = np.sum(hist * ((bin_centers - mean_angle) ** 2))
        directionality = 1.0 - variance / 1000.0  # Normalize
    else:
        directionality = 0.0
    
    return float(directionality), hist


def extract_tamura_features(image: np.ndarray) -> Dict[str, any]:
    """
    Extract all Tamura texture features.
    
    Args:
        image: Input image
    
    Returns:
        Dictionary containing:
            - 'coarseness': Coarseness value
            - 'contrast': Contrast value
            - 'directionality': Directionality value
            - 'direction_histogram': Direction histogram
    """
    coarseness = compute_tamura_coarseness(image)
    contrast = compute_tamura_contrast(image)
    directionality, dir_hist = compute_tamura_directionality(image)
    
    return {
        'coarseness': float(coarseness),
        'contrast': float(contrast),
        'directionality': float(directionality),
        'direction_histogram': dir_hist.tolist()
    }


def extract_texture_features(image: np.ndarray) -> Dict[str, any]:
    """
    Extract all texture features from an image.
    
    Args:
        image: Input image
    
    Returns:
        Dictionary containing:
            - 'gabor_features': Gabor filter responses
            - 'tamura_features': Tamura texture features
            - 'combined': Concatenated feature vector
    """
    # Extract Gabor features
    gabor_features = extract_gabor_features(image)
    
    # Extract Tamura features
    tamura_features = extract_tamura_features(image)
    
    # Combine into a single feature vector
    combined = np.concatenate([
        gabor_features,
        [tamura_features['coarseness']],
        [tamura_features['contrast']],
        [tamura_features['directionality']],
        tamura_features['direction_histogram']
    ])
    
    return {
        'gabor_features': gabor_features.tolist(),
        'tamura_coarseness': tamura_features['coarseness'],
        'tamura_contrast': tamura_features['contrast'],
        'tamura_directionality': tamura_features['directionality'],
        'tamura_direction_histogram': tamura_features['direction_histogram'],
        'combined': combined.tolist()
    }


if __name__ == "__main__":
    # Test the texture feature extraction
    print("Texture Feature Extraction Module")
    print("=" * 50)
    
    # Create a simple test image with texture
    test_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    features = extract_texture_features(test_img)
    
    print(f"Gabor Features: {len(features['gabor_features'])} values")
    print(f"Tamura Coarseness: {features['tamura_coarseness']:.4f}")
    print(f"Tamura Contrast: {features['tamura_contrast']:.4f}")
    print(f"Tamura Directionality: {features['tamura_directionality']:.4f}")
    print(f"Combined Texture Features: {len(features['combined'])} dimensions")
