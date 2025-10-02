"""
Vector Visualization Module for CBIR System
Provides graphical representation of image feature vectors and similarity metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

class VectorVisualizer:
    """Handles all visualization tasks for the CBIR system"""
    
    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor
        self.colors = {
            'query': '#FF6B6B',
            'cosine': '#4ECDC4', 
            'euclidean': '#45B7D1',
            'manhattan': '#96CEB4'
        }
        
    def reduce_dimensions(self, features, method='pca', n_components=2):
        """
        Reduce feature vector dimensions for visualization
        
        Args:
            features: numpy array of feature vectors
            method: 'pca' or 'tsne'
            n_components: number of dimensions to reduce to
        
        Returns:
            reduced features array
        """
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(features)-1))
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
            
        return reducer.fit_transform(features)
    
    def create_vector_scatter_plot(self, query_features, retrieved_features, 
                                 query_name="Query Image", retrieved_names=None,
                                 method_name="Similarity", reduction_method='pca'):
        """
        Create interactive scatter plot of feature vectors
        
        Args:
            query_features: Feature vector of query image
            retrieved_features: List of feature vectors for retrieved images
            query_name: Name/label for query image
            retrieved_names: List of names for retrieved images
            method_name: Name of similarity method used
            reduction_method: 'pca' or 'tsne' for dimensionality reduction
        
        Returns:
            Plotly figure object
        """
        # Combine all features for consistent dimensionality reduction
        all_features = np.vstack([query_features.reshape(1, -1)] + 
                               [feat.reshape(1, -1) for feat in retrieved_features])
        
        # Reduce dimensions
        reduced_features = self.reduce_dimensions(all_features, method=reduction_method)
        
        # Separate query and retrieved features
        query_reduced = reduced_features[0]
        retrieved_reduced = reduced_features[1:]
        
        # Create the plot
        fig = go.Figure()
        
        # Add query point
        fig.add_trace(go.Scatter(
            x=[query_reduced[0]], y=[query_reduced[1]],
            mode='markers+text',
            marker=dict(size=15, color=self.colors['query'], symbol='star'),
            text=[query_name],
            textposition="top center",
            name='Query Image',
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))
        
        # Add retrieved points
        for i, (reduced_feat, name) in enumerate(zip(retrieved_reduced, retrieved_names or [])):
            rank = i + 1
            display_name = name if name else f"Result {rank}"
            
            fig.add_trace(go.Scatter(
                x=[reduced_feat[0]], y=[reduced_feat[1]],
                mode='markers+text',
                marker=dict(size=12, color=self.colors.get(method_name.lower(), '#95A5A6')),
                text=[f"#{rank}"],
                textposition="top center",
                name=display_name,
                hovertemplate=f'<b>{display_name}</b><br>Rank: {rank}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Feature Vector Visualization ({reduction_method.upper()}) - {method_name}',
            xaxis_title=f'{reduction_method.upper()} Component 1',
            yaxis_title=f'{reduction_method.upper()} Component 2',
            width=800,
            height=600,
            showlegend=True,
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig
    
    def create_distance_visualization(self, query_features, retrieved_features, 
                                    distances, method_name, retrieved_names=None):
        """
        Create visualization showing distances/similarities between query and retrieved images
        
        Args:
            query_features: Query image feature vector
            retrieved_features: List of retrieved image feature vectors
            distances: List of distance/similarity scores
            method_name: Name of the similarity method
            retrieved_names: List of names for retrieved images
        
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distance/Similarity Scores', 'Feature Vector Comparison', 
                          'Cosine Angle Visualization', 'Distance Matrix Heatmap'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter", "colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Bar chart of distances/similarities
        x_labels = [f"#{i+1}" for i in range(len(distances))]
        if retrieved_names:
            x_labels = [f"#{i+1}: {name[:15]}..." if len(name) > 15 else f"#{i+1}: {name}" 
                       for i, name in enumerate(retrieved_names)]
        
        fig.add_trace(
            go.Bar(x=x_labels, y=distances, 
                   marker_color=self.colors.get(method_name.lower(), '#95A5A6'),
                   name=f'{method_name} Scores'),
            row=1, col=1
        )
        
        # 2. Feature vector magnitude comparison
        query_magnitude = np.linalg.norm(query_features)
        retrieved_magnitudes = [np.linalg.norm(feat) for feat in retrieved_features]
        
        fig.add_trace(
            go.Scatter(x=[0], y=[query_magnitude], 
                      mode='markers', marker=dict(size=15, color=self.colors['query'], symbol='star'),
                      name='Query Magnitude'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=list(range(1, len(retrieved_magnitudes)+1)), y=retrieved_magnitudes,
                      mode='markers+lines', 
                      marker=dict(size=10, color=self.colors.get(method_name.lower(), '#95A5A6')),
                      name='Retrieved Magnitudes'),
            row=1, col=2
        )
        
        # 3. Cosine angle visualization (if applicable)
        if method_name.lower() == 'cosine':
            angles = []
            for feat in retrieved_features:
                cos_sim = np.dot(query_features, feat) / (np.linalg.norm(query_features) * np.linalg.norm(feat))
                angle = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi
                angles.append(angle)
            
            fig.add_trace(
                go.Scatter(x=list(range(1, len(angles)+1)), y=angles,
                          mode='markers+lines',
                          marker=dict(size=10, color=self.colors['cosine']),
                          name='Cosine Angles (degrees)'),
                row=2, col=1
            )
        else:
            # For other methods, show normalized distances
            normalized_distances = np.array(distances) / np.max(distances) if np.max(distances) > 0 else distances
            fig.add_trace(
                go.Scatter(x=list(range(1, len(normalized_distances)+1)), y=normalized_distances,
                          mode='markers+lines',
                          marker=dict(size=10, color=self.colors.get(method_name.lower(), '#95A5A6')),
                          name='Normalized Distances'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text="Retrieved Images", row=1, col=1)
        fig.update_yaxes(title_text=f"{method_name} Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Image Index", row=1, col=2)
        fig.update_yaxes(title_text="Feature Vector Magnitude", row=1, col=2)
        
        fig.update_xaxes(title_text="Retrieved Images", row=2, col=1)
        if method_name.lower() == 'cosine':
            fig.update_yaxes(title_text="Angle (degrees)", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Normalized Distance", row=2, col=1)
        
        fig.update_layout(
            title=f'{method_name} Method - Distance and Similarity Analysis',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_multi_method_comparison(self, query_features, retrieved_results, method_names):
        """
        Create comparison visualization for multiple similarity methods
        
        Args:
            query_features: Query image feature vector
            retrieved_results: Dict with method names as keys and (features, distances, names) as values
            method_names: List of method names to compare
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Method Comparison - Scores', 'Vector Space Visualization (PCA)',
                          'Ranking Comparison', 'Score Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Colors for different methods
        method_colors = {
            'cosine': self.colors['cosine'],
            'euclidean': self.colors['euclidean'], 
            'manhattan': self.colors['manhattan']
        }
        
        # 1. Method comparison - scores for top 3 results
        top_k = 3
        x_labels = [f"Rank {i+1}" for i in range(top_k)]
        
        for method in method_names:
            if method in retrieved_results:
                _, distances, _ = retrieved_results[method]
                scores = distances[:top_k]
                fig.add_trace(
                    go.Bar(x=x_labels, y=scores, name=f'{method.title()}',
                          marker_color=method_colors.get(method, '#95A5A6')),
                    row=1, col=1
                )
        
        # 2. Vector space visualization using PCA
        all_features = [query_features]
        labels = ['Query']
        colors_list = [self.colors['query']]
        
        for method in method_names:
            if method in retrieved_results:
                features, _, names = retrieved_results[method]
                all_features.extend(features[:3])  # Top 3 for each method
                labels.extend([f'{method.title()}-{i+1}' for i in range(min(3, len(features)))])
                colors_list.extend([method_colors.get(method, '#95A5A6')] * min(3, len(features)))
        
        if len(all_features) > 1:
            all_features_array = np.vstack([feat.reshape(1, -1) for feat in all_features])
            reduced_features = self.reduce_dimensions(all_features_array, method='pca')
            
            fig.add_trace(
                go.Scatter(x=reduced_features[:, 0], y=reduced_features[:, 1],
                          mode='markers+text',
                          marker=dict(size=10, color=colors_list),
                          text=labels,
                          textposition="top center",
                          name='Feature Vectors'),
                row=1, col=2
            )
        
        # 3. Ranking comparison
        rank_data = {}
        for method in method_names:
            if method in retrieved_results:
                _, distances, names = retrieved_results[method]
                rank_data[method] = list(range(1, min(6, len(distances)+1)))  # Top 5 ranks
        
        for method, ranks in rank_data.items():
            fig.add_trace(
                go.Bar(x=[f"Top {i}" for i in ranks], y=ranks, name=f'{method.title()} Ranking',
                      marker_color=method_colors.get(method, '#95A5A6')),
                row=2, col=1
            )
        
        # 4. Score distribution
        for method in method_names:
            if method in retrieved_results:
                _, distances, _ = retrieved_results[method]
                fig.add_trace(
                    go.Box(y=distances, name=f'{method.title()}',
                          marker_color=method_colors.get(method, '#95A5A6')),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title='Multi-Method CBIR Analysis Dashboard',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_feature_heatmap(self, query_features, retrieved_features, retrieved_names=None):
        """
        Create heatmap visualization of feature vectors
        
        Args:
            query_features: Query image feature vector
            retrieved_features: List of retrieved image feature vectors
            retrieved_names: List of names for retrieved images
        
        Returns:
            Plotly figure object
        """
        # Combine all features
        all_features = np.vstack([query_features.reshape(1, -1)] + 
                               [feat.reshape(1, -1) for feat in retrieved_features])
        
        # Create labels
        labels = ['Query Image']
        if retrieved_names:
            labels.extend([f"#{i+1}: {name}" for i, name in enumerate(retrieved_names)])
        else:
            labels.extend([f"Result #{i+1}" for i in range(len(retrieved_features))])
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=all_features,
            y=labels,
            colorscale='Viridis',
            hovertemplate='Feature %{x}<br>%{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Vector Heatmap Comparison',
            xaxis_title='Feature Dimensions',
            yaxis_title='Images',
            height=400 + len(labels) * 30,
            template='plotly_white'
        )
        
        return fig
    
    def get_image_thumbnail(self, image_path, size=(100, 100)):
        """
        Create thumbnail of image for visualization
        
        Args:
            image_path: Path to image file
            size: Tuple of (width, height) for thumbnail
        
        Returns:
            Base64 encoded image string
        """
        try:
            image = Image.open(image_path)
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            return None
    
    def create_similarity_network(self, query_features, retrieved_features, distances, 
                                retrieved_names=None, threshold=0.5):
        """
        Create network graph showing similarity relationships
        
        Args:
            query_features: Query image feature vector
            retrieved_features: List of retrieved image feature vectors  
            distances: List of similarity scores
            retrieved_names: List of names for retrieved images
            threshold: Similarity threshold for connecting nodes
        
        Returns:
            Plotly figure object
        """
        # This is a placeholder for advanced network visualization
        # Could be implemented with networkx and plotly for complex similarity networks
        pass

def create_visualization_summary(visualizer, query_features, method_results, method_names):
    """
    Create a comprehensive visualization summary for the CBIR results
    
    Args:
        visualizer: VectorVisualizer instance
        query_features: Query image feature vector
        method_results: Dict with method results
        method_names: List of selected methods
    
    Returns:
        Dict of plotly figures
    """
    figures = {}
    
    # Individual method visualizations
    for method in method_names:
        if method in method_results:
            features, distances, names = method_results[method]
            
            # Vector scatter plot
            figures[f'{method}_vector_plot'] = visualizer.create_vector_scatter_plot(
                query_features, features, retrieved_names=names, method_name=method
            )
            
            # Distance visualization
            figures[f'{method}_distance_plot'] = visualizer.create_distance_visualization(
                query_features, features, distances, method, names
            )
    
    # Multi-method comparison
    if len(method_names) > 1:
        figures['multi_method_comparison'] = visualizer.create_multi_method_comparison(
            query_features, method_results, method_names
        )
    
    # Feature heatmap
    if method_names:
        primary_method = method_names[0]
        if primary_method in method_results:
            features, _, names = method_results[primary_method]
            figures['feature_heatmap'] = visualizer.create_feature_heatmap(
                query_features, features, names
            )
    
    return figures