"""
Test script to verify the installation and basic functionality.

This script tests the feature extraction modules with sample images.
"""

import numpy as np
import cv2
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extraction.shape_features import extract_shape_features
from feature_extraction.texture_features import extract_texture_features


def create_test_images():
    """Create some simple test images."""
    images = {}
    
    # 1. Circle
    img_circle = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img_circle, (100, 100), 60, 255, -1)
    images['circle'] = img_circle
    
    # 2. Rectangle
    img_rect = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img_rect, (50, 50), (150, 150), 255, -1)
    images['rectangle'] = img_rect
    
    # 3. Triangle
    img_tri = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array([[100, 40], [40, 160], [160, 160]], np.int32)
    cv2.fillPoly(img_tri, [pts], 255)
    images['triangle'] = img_tri
    
    # 4. Random texture
    np.random.seed(42)
    img_texture = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    images['texture'] = img_texture
    
    return images


def test_shape_features():
    """Test shape feature extraction."""
    print("=" * 70)
    print("Testing Shape Feature Extraction")
    print("=" * 70)
    
    images = create_test_images()
    
    for name, img in images.items():
        print(f"\nTesting with {name} image...")
        
        try:
            features = extract_shape_features(img)
            
            print(f"  ✓ Fourier descriptors: {len(features['fourier_descriptors'])} values")
            print(f"  ✓ Direction histogram: {len(features['direction_histogram'])} bins")
            print(f"  ✓ Combined shape features: {len(features['combined'])} dimensions")
            print(f"  Sample values: {features['fourier_descriptors'][:3]}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return False
    
    print("\n✓ Shape feature extraction working correctly!")
    return True


def test_texture_features():
    """Test texture feature extraction."""
    print("\n" + "=" * 70)
    print("Testing Texture Feature Extraction")
    print("=" * 70)
    
    images = create_test_images()
    
    for name, img in images.items():
        print(f"\nTesting with {name} image...")
        
        try:
            features = extract_texture_features(img)
            
            print(f"  ✓ Gabor features: {len(features['gabor_features'])} values")
            print(f"  ✓ Tamura coarseness: {features['tamura_coarseness']:.4f}")
            print(f"  ✓ Tamura contrast: {features['tamura_contrast']:.4f}")
            print(f"  ✓ Tamura directionality: {features['tamura_directionality']:.4f}")
            print(f"  ✓ Combined texture features: {len(features['combined'])} dimensions")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return False
    
    print("\n✓ Texture feature extraction working correctly!")
    return True


def test_combined_features():
    """Test combined feature extraction."""
    print("\n" + "=" * 70)
    print("Testing Combined Feature Extraction")
    print("=" * 70)
    
    img = create_test_images()['circle']
    
    try:
        shape_features = extract_shape_features(img)
        texture_features = extract_texture_features(img)
        
        combined = shape_features['combined'] + texture_features['combined']
        
        print(f"\nFeature dimensions:")
        print(f"  - Shape features: {len(shape_features['combined'])}")
        print(f"  - Texture features: {len(texture_features['combined'])}")
        print(f"  - Combined features: {len(combined)}")
        
        print("\n✓ Combined feature extraction working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("IMAGE RETRIEVAL SYSTEM - TEST SUITE")
    print("=" * 70)
    
    # Check imports
    print("\nChecking dependencies...")
    try:
        import cv2
        import numpy
        import scipy
        import matplotlib
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return 1
    
    # Run tests
    results = []
    
    results.append(("Shape Features", test_shape_features()))
    results.append(("Texture Features", test_texture_features()))
    results.append(("Combined Features", test_combined_features()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Place your images in a folder")
        print("2. Run: python a_extract_features.py --folder <path_to_folder>")
        print("3. Run: python b_image_search.py --folder <path_to_folder> --query <image_name>")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
