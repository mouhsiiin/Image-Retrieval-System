"""
Generate all figures for the report.

This script creates visualizations that you can include in your report.
Make sure to create a 'figures' folder first, or the script will create it.
"""

import os
from visualize_results import *

# Create figures folder if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')
    print("Created 'figures' folder")

print("=" * 70)
print("GENERATING FIGURES FOR REPORT")
print("=" * 70)

try:
    # Figure 1: Contour extraction process - Apple
    print("\n[1/9] Creating Figure 1: Contour Extraction (Apple)...")
    visualize_contour_extraction(
        'Formes/apple-1.gif', 
        'figures/fig1_contour_apple.png'
    )
    
    # Figure 2: Contour extraction process - Bell  
    print("\n[2/9] Creating Figure 2: Contour Extraction (Bell)...")
    visualize_contour_extraction(
        'Formes/bell-1.gif',
        'figures/fig2_contour_bell.png'
    )
    
    # Figure 3: Contour extraction process - Bird
    print("\n[3/9] Creating Figure 3: Contour Extraction (Bird)...")
    visualize_contour_extraction(
        'Formes/bird-7.gif',
        'figures/fig3_contour_bird.png'
    )
    
    # Figure 4: Gabor filters on texture
    print("\n[4/9] Creating Figure 4: Gabor Filters (Texture)...")
    visualize_gabor_filters(
        'Textures/Im01.jpg',
        'figures/fig4_gabor_texture.png'
    )
    
    # Figure 5: Gabor filters on another texture
    print("\n[5/9] Creating Figure 5: Gabor Filters (Another Texture)...")
    visualize_gabor_filters(
        'Textures/Im15.jpg',
        'figures/fig5_gabor_texture2.png'
    )
    
    # Figure 6: Shape feature comparison
    print("\n[6/9] Creating Figure 6: Shape Feature Comparison...")
    print("Please screenshot the matplotlib window that appears!")
    visualize_feature_comparison([
        'Formes/apple-1.json',
        'Formes/bell-1.json',
        'Formes/bird-7.json'
    ], feature_type='shape')
    
    # Figure 7: Texture feature comparison
    print("\n[7/9] Creating Figure 7: Texture Feature Comparison...")
    print("Please screenshot the matplotlib window that appears!")
    visualize_feature_comparison([
        'Textures/Im01.json',
        'Textures/Im10.json',
        'Textures/Im20.json'
    ], feature_type='texture')
    
    # Figure 8: Distance matrix for shapes
    print("\n[8/9] Creating Figure 8: Distance Matrix (Shapes)...")
    print("Please screenshot the matplotlib window that appears!")
    plot_distance_matrix('Formes', max_images=15)
    
    # Figure 9: Distance matrix for textures
    print("\n[9/9] Creating Figure 9: Distance Matrix (Textures)...")
    print("Please screenshot the matplotlib window that appears!")
    plot_distance_matrix('Textures', max_images=15)
    
    print("\n" + "=" * 70)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files in 'figures/' folder:")
    print("  - fig1_contour_apple.png")
    print("  - fig2_contour_bell.png")
    print("  - fig3_contour_bird.png")
    print("  - fig4_gabor_texture.png")
    print("  - fig5_gabor_texture2.png")
    print("\nInteractive figures displayed (screenshot them!):")
    print("  - Figure 6: Shape Feature Comparison")
    print("  - Figure 7: Texture Feature Comparison")
    print("  - Figure 8: Distance Matrix (Shapes)")
    print("  - Figure 9: Distance Matrix (Textures)")
    print("\nYou can now include these figures in your report!")
    
except FileNotFoundError as e:
    print(f"\n✗ Error: File not found - {e}")
    print("Make sure you have run feature extraction first:")
    print("  python a_extract_features.py --folder Formes")
    print("  python a_extract_features.py --folder Textures")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
