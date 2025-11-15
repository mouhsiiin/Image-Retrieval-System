"""
Visualization utilities for creating figures for the report.

This module provides functions to visualize features and results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os


def visualize_contour_extraction(image_path: str, output_path: str = None):
    """
    Visualize the contour extraction process.
    
    Args:
        image_path: Path to the image
        output_path: Optional path to save the figure
    """
    from feature_extraction.shape_features import find_largest_contour
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contour
    contour = find_largest_contour(image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grayscale
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale')
    axes[1].axis('off')
    
    # Binary
    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title('Binary Threshold')
    axes[2].axis('off')
    
    # Contour
    contour_img = image.copy()
    if contour is not None:
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
    axes[3].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f'Extracted Contour\n({len(contour) if contour is not None else 0} points)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()


def visualize_gabor_filters(image_path: str, output_path: str = None):
    """
    Visualize Gabor filter responses.
    
    Args:
        image_path: Path to the image
        output_path: Optional path to save the figure
    """
    from feature_extraction.texture_features import create_gabor_kernels
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    image = image.astype(np.float32) / 255.0
    
    # Create kernels
    kernels = create_gabor_kernels()
    
    # Apply filters
    num_kernels = min(12, len(kernels))  # Show first 12
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(num_kernels):
        filtered = cv2.filter2D(image, cv2.CV_32F, kernels[i])
        axes[i].imshow(filtered, cmap='gray')
        axes[i].set_title(f'Gabor Filter {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Gabor Filter Responses', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()


def visualize_feature_comparison(json_paths: list, feature_type: str = 'all'):
    """
    Compare features across multiple images.
    
    Args:
        json_paths: List of paths to JSON feature files
        feature_type: 'shape', 'texture', or 'all'
    """
    features_list = []
    names = []
    
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            features_list.append(data)
            names.append(Path(json_path).stem)
    
    if feature_type == 'shape' or feature_type == 'all':
        # Plot Fourier descriptors
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for features, name in zip(features_list, names):
            fourier = features['shape_features']['fourier_descriptors']
            ax.plot(fourier, marker='o', label=name, linewidth=2)
        
        ax.set_xlabel('Coefficient Index')
        ax.set_ylabel('Magnitude')
        ax.set_title('Fourier Descriptors Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot direction histograms
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(8)
        width = 0.15
        
        for i, (features, name) in enumerate(zip(features_list, names)):
            direction_hist = features['shape_features']['direction_histogram']
            ax.bar(x + i * width, direction_hist, width, label=name)
        
        ax.set_xlabel('Direction Bin')
        ax.set_ylabel('Normalized Frequency')
        ax.set_title('Contour Direction Histogram Comparison')
        ax.set_xticks(x + width * len(features_list) / 2)
        ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    if feature_type == 'texture' or feature_type == 'all':
        # Plot Tamura features
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        tamura_features = ['tamura_coarseness', 'tamura_contrast', 'tamura_directionality']
        titles = ['Coarseness', 'Contrast', 'Directionality']
        
        for ax, feature_name, title in zip(axes, tamura_features, titles):
            values = [f['texture_features'][feature_name] for f in features_list]
            ax.bar(names, values, color='steelblue')
            ax.set_title(f'Tamura {title}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


def plot_distance_matrix(folder_path: str, max_images: int = 10):
    """
    Create a distance matrix heatmap for images in a folder.
    
    Args:
        folder_path: Path to folder containing JSON files
        max_images: Maximum number of images to include
    """
    from b_image_search import compute_euclidean_distance
    
    # Get JSON files
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    json_files = json_files[:max_images]
    
    # Load features
    features_list = []
    names = []
    
    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            features_list.append(data['combined_feature_vector'])
            names.append(Path(json_file).stem)
    
    # Compute distance matrix
    n = len(features_list)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = compute_euclidean_distance(
                    features_list[i], 
                    features_list[j]
                )
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(distance_matrix, cmap='YlOrRd')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Distance', rotation=-90, va="bottom")
    
    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{distance_matrix[i, j]:.1f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title("Image Distance Matrix (Euclidean)")
    plt.tight_layout()
    plt.show()


def create_retrieval_result_figure(query_image_path: str, 
                                   similar_images: list,
                                   folder_path: str,
                                   output_path: str = None):
    """
    Create a professional figure showing retrieval results for the report.
    
    Args:
        query_image_path: Path to query image
        similar_images: List of (image_name, distance) tuples
        folder_path: Folder containing images
        output_path: Optional path to save figure
    """
    num_results = len(similar_images)
    
    # Create figure with better layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # Query image (larger, top-left)
    ax_query = fig.add_subplot(gs[0, :2])
    query_img = cv2.imread(query_image_path)
    if query_img is not None:
        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        ax_query.imshow(query_img_rgb)
        ax_query.set_title('QUERY IMAGE\n' + os.path.basename(query_image_path),
                          fontsize=14, fontweight='bold', color='red')
        ax_query.axis('off')
    
    # Similar images
    positions = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    
    for idx, (image_name, distance) in enumerate(similar_images[:6]):
        if idx < len(positions):
            row, col = positions[idx]
            ax = fig.add_subplot(gs[row, col])
            
            image_path = os.path.join(folder_path, image_name)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    
                    # Color code by rank
                    colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'lavender']
                    border_color = colors[idx] if idx < len(colors) else 'gray'
                    
                    ax.set_title(f'Rank #{idx+1}\n{image_name}\nDist: {distance:.4f}',
                               fontsize=10, bbox=dict(boxstyle='round', 
                                                     facecolor=border_color, 
                                                     alpha=0.5))
                    ax.axis('off')
    
    plt.suptitle('Content-Based Image Retrieval Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization Utilities for Image Retrieval System")
    print("=" * 70)
    print("\nAvailable functions:")
    print("1. visualize_contour_extraction(image_path)")
    print("2. visualize_gabor_filters(image_path)")
    print("3. visualize_feature_comparison([json_paths])")
    print("4. plot_distance_matrix(folder_path)")
    print("5. create_retrieval_result_figure(query, results, folder)")
    print("\nExample usage:")
    print("  from visualize_results import visualize_contour_extraction")
    print("  visualize_contour_extraction('path/to/image.jpg', 'output.png')")
