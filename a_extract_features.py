"""
Program (a): Feature Extraction to JSON

This program processes all images in a folder and extracts both shape and 
texture features, saving them to JSON files with the same name as each image.

Usage:
    python a_extract_features.py --folder <path_to_image_folder>
    
Example:
    python a_extract_features.py --folder ./images/dataset1
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
from feature_extraction.shape_features import extract_shape_features
from feature_extraction.texture_features import extract_texture_features


def is_image_file(filename: str) -> bool:
    """
    Check if a file is an image based on its extension.
    
    Args:
        filename: Name of the file
    
    Returns:
        True if the file is an image
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    ext = Path(filename).suffix.lower()
    return ext in valid_extensions


def extract_all_features(image_path: str) -> Dict:
    """
    Extract all features (shape + texture) from an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary containing all extracted features
    """
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Extract shape features
    print(f"  - Extracting shape features...")
    shape_features = extract_shape_features(image)
    
    # Extract texture features
    print(f"  - Extracting texture features...")
    texture_features = extract_texture_features(image)
    
    # Combine all features
    all_features = {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'shape_features': shape_features,
        'texture_features': texture_features,
        'combined_feature_vector': (
            shape_features['combined'] + 
            texture_features['combined']
        )
    }
    
    return all_features


def process_folder(folder_path: str, output_folder: str = None) -> List[str]:
    """
    Process all images in a folder and save features to JSON files.
    
    Args:
        folder_path: Path to the folder containing images
        output_folder: Optional output folder for JSON files (default: same as input)
    
    Returns:
        List of processed image paths
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Use input folder as output folder if not specified
    if output_folder is None:
        output_folder = folder_path
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_files = [
        f for f in os.listdir(folder_path) 
        if is_image_file(f)
    ]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    print("=" * 70)
    
    processed_files = []
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        
        try:
            # Extract features
            features = extract_all_features(image_path)
            
            # Create JSON filename
            json_filename = Path(image_file).stem + '.json'
            json_path = os.path.join(output_folder, json_filename)
            
            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Features saved to: {json_filename}")
            print(f"  - Shape feature dimensions: {len(features['shape_features']['combined'])}")
            print(f"  - Texture feature dimensions: {len(features['texture_features']['combined'])}")
            print(f"  - Total feature dimensions: {len(features['combined_feature_vector'])}")
            
            processed_files.append(image_path)
            
        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {str(e)}")
            continue
    
    print("\n" + "=" * 70)
    print(f"Processing complete! {len(processed_files)}/{len(image_files)} images processed successfully.")
    
    return processed_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract features from all images in a folder and save to JSON files.'
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to the folder containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output folder for JSON files (default: same as input folder)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Program (a): Feature Extraction to JSON")
    print("=" * 70)
    print(f"Input folder: {args.folder}")
    print(f"Output folder: {args.output or args.folder}")
    print()
    
    try:
        processed_files = process_folder(args.folder, args.output)
        
        if processed_files:
            print(f"\n✓ Successfully processed {len(processed_files)} images!")
            print(f"JSON files saved in: {args.output or args.folder}")
        else:
            print("\n✗ No images were processed.")
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
