"""
Test script to verify the visualization module functionality
"""
import numpy as np
import sys
import os
sys.path.append('/workspaces/Computer-Vision-AAST-2025/cbir_system')

from visualizer import VectorVisualizer
from feature_extractor import FeatureExtractor
import matplotlib.pyplot as plt

def test_visualizations():
    """Test basic visualization functionality"""
    print("Testing Vector Visualization Module...")
    
    # Initialize visualizer
    visualizer = VectorVisualizer()
    
    # Create sample data
    np.random.seed(42)
    
    # Simulate query features (128-dimensional)
    query_features = np.random.rand(128)
    
    # Simulate retrieved features (3 results)
    retrieved_features = [
        np.random.rand(128) + 0.1,  # Similar to query
        np.random.rand(128) + 0.5,  # Moderately similar
        np.random.rand(128) + 1.0   # Less similar
    ]
    
    # Simulate distances
    cosine_distances = [0.95, 0.78, 0.62]
    euclidean_distances = [2.1, 5.4, 8.9]
    manhattan_distances = [12.3, 28.7, 45.2]
    
    retrieved_names = ["Pink Primrose", "Tiger Lily", "Bird of Paradise"]
    
    print("✓ Sample data created")
    
    # Test 1: Vector scatter plot
    try:
        fig1 = visualizer.create_vector_scatter_plot(
            query_features, retrieved_features, 
            retrieved_names=retrieved_names, 
            method_name="cosine",
            reduction_method='pca'
        )
        print("✓ Vector scatter plot creation - SUCCESS")
    except Exception as e:
        print(f"✗ Vector scatter plot creation - FAILED: {e}")
    
    # Test 2: Distance visualization
    try:
        fig2 = visualizer.create_distance_visualization(
            query_features, retrieved_features, 
            cosine_distances, "cosine", retrieved_names
        )
        print("✓ Distance visualization creation - SUCCESS")
    except Exception as e:
        print(f"✗ Distance visualization creation - FAILED: {e}")
    
    # Test 3: Multi-method comparison
    try:
        method_results = {
            'cosine': (retrieved_features, cosine_distances, retrieved_names),
            'euclidean': (retrieved_features, euclidean_distances, retrieved_names),
            'manhattan': (retrieved_features, manhattan_distances, retrieved_names)
        }
        
        fig3 = visualizer.create_multi_method_comparison(
            query_features, method_results, ['cosine', 'euclidean', 'manhattan']
        )
        print("✓ Multi-method comparison - SUCCESS")
    except Exception as e:
        print(f"✗ Multi-method comparison - FAILED: {e}")
    
    # Test 4: Feature heatmap
    try:
        fig4 = visualizer.create_feature_heatmap(
            query_features, retrieved_features, retrieved_names
        )
        print("✓ Feature heatmap creation - SUCCESS")
    except Exception as e:
        print(f"✗ Feature heatmap creation - FAILED: {e}")
    
    # Test 5: Dimensionality reduction
    try:
        all_features = np.vstack([query_features.reshape(1, -1)] + 
                               [feat.reshape(1, -1) for feat in retrieved_features])
        
        pca_result = visualizer.reduce_dimensions(all_features, method='pca')
        print(f"✓ PCA reduction - SUCCESS (shape: {pca_result.shape})")
        
        # Test t-SNE with minimum required samples
        if len(all_features) >= 4:  # t-SNE needs at least 4 samples
            tsne_result = visualizer.reduce_dimensions(all_features, method='tsne')
            print(f"✓ t-SNE reduction - SUCCESS (shape: {tsne_result.shape})")
        else:
            print("⚠ t-SNE reduction - SKIPPED (insufficient samples)")
            
    except Exception as e:
        print(f"✗ Dimensionality reduction - FAILED: {e}")
    
    print("\nVisualization module test completed!")
    return True

if __name__ == "__main__":
    test_visualizations()