"""
Shape Feature Extraction Module

This module implements shape-based descriptors:
1. Fourier Transform Coefficients on contours
2. Histogram of Contour Directions
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


def find_largest_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the contour with the most points in the image.
    
    Args:
        image: Input image (grayscale or color)
    
    Returns:
        The largest contour or None if no contours found
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # Return contour with most points
    largest_contour = max(contours, key=len)
    return largest_contour


def compute_fourier_descriptors(contour: np.ndarray, num_descriptors: int = 20) -> np.ndarray:
    """
    Compute Fourier descriptors from a contour.
    
    The Fourier descriptors are obtained by:
    1. Treating the contour as a complex function: z(t) = x(t) + j*y(t)
    2. Computing the FFT
    3. Keeping the first N coefficients (normalized and translation/rotation invariant)
    
    Args:
        contour: Contour points array of shape (N, 1, 2)
        num_descriptors: Number of Fourier descriptors to keep
    
    Returns:
        Array of Fourier descriptor magnitudes
    """
    # Reshape contour to (N, 2)
    contour = contour.reshape(-1, 2)
    
    # Convert to complex representation
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    
    # Compute FFT
    fourier_result = np.fft.fft(complex_contour)
    
    # Take magnitude and normalize
    fourier_magnitudes = np.abs(fourier_result)
    
    # Make translation invariant (remove DC component)
    fourier_magnitudes[0] = 0
    
    # Make scale invariant (normalize by first non-zero coefficient)
    if fourier_magnitudes[1] != 0:
        fourier_magnitudes = fourier_magnitudes / fourier_magnitudes[1]
    
    # Return first num_descriptors coefficients
    descriptors = fourier_magnitudes[:num_descriptors]
    
    return descriptors


def compute_contour_direction_histogram(contour: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """
    Compute histogram of contour directions.
    
    This descriptor captures the distribution of edge orientations along the contour.
    
    Args:
        contour: Contour points array of shape (N, 1, 2)
        num_bins: Number of histogram bins (typically 8 for 8 directions)
    
    Returns:
        Normalized histogram of contour directions
    """
    # Reshape contour
    contour = contour.reshape(-1, 2)
    
    if len(contour) < 2:
        return np.zeros(num_bins)
    
    # Compute direction vectors between consecutive points
    directions = np.diff(contour, axis=0)
    
    # Compute angles (in radians)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    
    # Convert to degrees and normalize to [0, 360)
    angles_deg = np.degrees(angles) % 360
    
    # Create histogram
    hist, _ = np.histogram(angles_deg, bins=num_bins, range=(0, 360))
    
    # Normalize histogram
    hist = hist.astype(np.float32)
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist


def extract_shape_features(image: np.ndarray, 
                          num_fourier: int = 20, 
                          num_direction_bins: int = 8) -> Dict[str, np.ndarray]:
    """
    Extract all shape features from an image.
    
    Args:
        image: Input image
        num_fourier: Number of Fourier descriptors
        num_direction_bins: Number of bins for direction histogram
    
    Returns:
        Dictionary containing:
            - 'fourier_descriptors': Fourier transform coefficients
            - 'direction_histogram': Histogram of contour directions
            - 'combined': Concatenated feature vector
    """
    # Find largest contour
    contour = find_largest_contour(image)
    
    if contour is None or len(contour) < 10:
        # Return zero features if no valid contour found
        fourier_desc = np.zeros(num_fourier)
        direction_hist = np.zeros(num_direction_bins)
    else:
        # Compute Fourier descriptors
        fourier_desc = compute_fourier_descriptors(contour, num_fourier)
        
        # Compute direction histogram
        direction_hist = compute_contour_direction_histogram(contour, num_direction_bins)
    
    # Combine all shape features
    combined = np.concatenate([fourier_desc, direction_hist])
    
    return {
        'fourier_descriptors': fourier_desc.tolist(),
        'direction_histogram': direction_hist.tolist(),
        'combined': combined.tolist()
    }


if __name__ == "__main__":
    # Test the shape feature extraction
    print("Shape Feature Extraction Module")
    print("=" * 50)
    
    # Create a simple test image with a circle
    test_img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(test_img, (100, 100), 50, 255, -1)
    
    features = extract_shape_features(test_img)
    
    print(f"Fourier Descriptors: {len(features['fourier_descriptors'])} values")
    print(f"Direction Histogram: {len(features['direction_histogram'])} bins")
    print(f"Combined Shape Features: {len(features['combined'])} dimensions")
    print("\nSample Fourier Descriptors (first 5):", features['fourier_descriptors'][:5])
    print("Direction Histogram:", features['direction_histogram'])
