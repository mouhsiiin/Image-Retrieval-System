"""
Program (b): Image Similarity Search

This program takes a query image and finds the 6 most similar images from a 
folder based on their feature vectors (previously extracted and saved as JSON).

The program displays the results with distance values as captions.

Usage:
    python b_image_search.py --folder <path_to_image_folder> --query <query_image_name>
    
Example:
    python b_image_search.py --folder ./images/dataset1 --query image001.jpg
"""

import os
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def load_features_from_json(json_path: str) -> Dict:
    """
    Load features from a JSON file.
    
    Args:
        json_path: Path to the JSON file
    
    Returns:
        Dictionary containing features
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        features = json.load(f)
    return features


def compute_euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Compute Euclidean distance between two feature vectors.
    
    Args:
        vector1: First feature vector
        vector2: Second feature vector
    
    Returns:
        Euclidean distance
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    return float(np.linalg.norm(v1 - v2))


def compute_cosine_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Compute cosine distance between two feature vectors.
    Cosine distance = 1 - cosine similarity
    
    Args:
        vector1: First feature vector
        vector2: Second feature vector
    
    Returns:
        Cosine distance (0 = identical, 2 = opposite)
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    # Compute cosine similarity
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norm_product == 0:
        return 1.0
    
    cosine_similarity = dot_product / norm_product
    # Convert to distance (0 = most similar)
    cosine_distance = 1 - cosine_similarity
    
    return float(cosine_distance)


def find_similar_images(query_image_name: str, 
                       folder_path: str, 
                       top_k: int = 6,
                       distance_metric: str = 'euclidean') -> List[Tuple[str, float]]:
    """
    Find the most similar images to a query image.
    
    Args:
        query_image_name: Name of the query image
        folder_path: Path to the folder containing images and JSON files
        top_k: Number of similar images to return
        distance_metric: Distance metric to use ('euclidean' or 'cosine')
    
    Returns:
        List of tuples (image_name, distance) sorted by similarity
    """
    # Load query features
    query_json = Path(query_image_name).stem + '.json'
    query_json_path = os.path.join(folder_path, query_json)
    
    if not os.path.exists(query_json_path):
        raise ValueError(f"Features not found for query image: {query_json}")
    
    query_features = load_features_from_json(query_json_path)
    query_vector = query_features['combined_feature_vector']
    
    print(f"Query image: {query_image_name}")
    print(f"Feature vector size: {len(query_vector)}")
    print(f"Distance metric: {distance_metric}")
    print()
    
    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"No JSON feature files found in {folder_path}")
    
    print(f"Comparing with {len(json_files)} images...")
    
    # Compute distances
    distances = []
    
    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        
        try:
            features = load_features_from_json(json_path)
            image_name = features['image_name']
            
            # Skip the query image itself
            if image_name == query_image_name:
                continue
            
            feature_vector = features['combined_feature_vector']
            
            # Compute distance
            if distance_metric == 'euclidean':
                distance = compute_euclidean_distance(query_vector, feature_vector)
            elif distance_metric == 'cosine':
                distance = compute_cosine_distance(query_vector, feature_vector)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")
            
            distances.append((image_name, distance))
            
        except Exception as e:
            print(f"Warning: Error processing {json_file}: {str(e)}")
            continue
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])
    
    # Return top k
    return distances[:top_k]


def display_results(query_image_path: str, 
                   similar_images: List[Tuple[str, float]], 
                   folder_path: str):
    """
    Display the query image and the most similar images with distances.
    
    Args:
        query_image_path: Path to the query image
        similar_images: List of (image_name, distance) tuples
        folder_path: Folder containing the images
    """
    # Create figure
    num_images = len(similar_images) + 1  # +1 for query image
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle('Image Similarity Search Results', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Display query image
    query_img = cv2.imread(query_image_path)
    if query_img is not None:
        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(query_img_rgb)
        axes[0].set_title('QUERY IMAGE\n' + os.path.basename(query_image_path), 
                         fontweight='bold', fontsize=12, color='red')
        axes[0].axis('off')
    
    # Display similar images
    for i, (image_name, distance) in enumerate(similar_images, 1):
        image_path = os.path.join(folder_path, image_name)
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(f'#{i}: {image_name}\nDistance: {distance:.4f}', 
                                fontsize=10)
                axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def print_results(similar_images: List[Tuple[str, float]]):
    """
    Print results to console.
    
    Args:
        similar_images: List of (image_name, distance) tuples
    """
    print("\n" + "=" * 70)
    print("SIMILARITY SEARCH RESULTS")
    print("=" * 70)
    
    for i, (image_name, distance) in enumerate(similar_images, 1):
        print(f"{i}. {image_name:<40} Distance: {distance:.6f}")
    
    print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Find similar images based on extracted features.'
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to the folder containing images and JSON feature files'
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Name of the query image'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=6,
        help='Number of similar images to retrieve (default: 6)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        choices=['euclidean', 'cosine'],
        help='Distance metric to use (default: euclidean)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display images (only print results)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Program (b): Image Similarity Search")
    print("=" * 70)
    print()
    
    try:
        # Find similar images
        similar_images = find_similar_images(
            args.query,
            args.folder,
            args.top_k,
            args.metric
        )
        
        if not similar_images:
            print("No similar images found.")
            return 1
        
        # Print results
        print_results(similar_images)
        
        # Display results
        if not args.no_display:
            query_image_path = os.path.join(args.folder, args.query)
            if os.path.exists(query_image_path):
                display_results(query_image_path, similar_images, args.folder)
            else:
                print(f"\nWarning: Query image not found at {query_image_path}")
                print("Results printed above.")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
