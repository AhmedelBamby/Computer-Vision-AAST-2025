"""
Streamlit Web Application for CBIR (Content-Based Image Retrieval) System
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extractor import FeatureExtractor
from similarity_calculator import CBIRSystem

# Configure Streamlit page
st.set_page_config(
    page_title="CBIR System - Flower Image Retrieval",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .method-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .similarity-score {
        font-weight: bold;
        color: #28a745;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cbir_system():
    """Load the CBIR system with cached features database"""
    try:
        features_db_path = "features/features_db.npy"
        if Path(features_db_path).exists():
            return CBIRSystem(features_db_path)
        else:
            st.error("Features database not found. Please run feature extraction first.")
            return None
    except Exception as e:
        st.error(f"Error loading CBIR system: {e}")
        return None

@st.cache_resource
def load_feature_extractor():
    """Load feature extractor"""
    return FeatureExtractor()

def process_uploaded_image(uploaded_file):
    """Process the uploaded image and extract features"""
    try:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to numpy array
        image_array = np.array(image)
        
        # Save temporarily for feature extraction
        temp_path = "temp_query_image.jpg"
        image.save(temp_path)
        
        return image_array, temp_path
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def display_similarity_results(results, method_name, dataset_path):
    """Display similarity results for a specific method"""
    st.markdown(f"### {method_name.title()} Results")
    
    if not results:
        st.warning(f"No results found for {method_name}")
        return
    
    # Create columns for displaying results
    cols = st.columns(3)
    
    for idx, (image_name, score) in enumerate(results):
        col = cols[idx % 3]
        
        with col:
            # Handle the new dataset structure (category/image.jpg)
            image_path = Path(dataset_path) / image_name
            if image_path.exists():
                try:
                    image = Image.open(image_path)
                    # Resize for consistent display
                    image = image.resize((200, 200))
                    st.image(image, caption=f"Rank {idx + 1}", width=200)
                    
                    # Display similarity score
                    if method_name == 'cosine':
                        st.markdown(f'<p class="similarity-score">Cosine Similarity: {score:.4f}</p>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="similarity-score">{method_name.title()} Distance: {score:.4f}</p>', 
                                  unsafe_allow_html=True)
                    
                    # Extract category and image info
                    parts = Path(image_name).parts
                    if len(parts) >= 2:
                        category = parts[0]
                        image_file = parts[1]
                        st.markdown(f"**Category:** {category}")
                        st.markdown(f"**Image:** {image_file}")
                    else:
                        st.markdown(f"**Image:** {image_name}")
                        
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.error(f"Image not found: {image_name}")

def create_comparison_chart(results, selected_methods):
    """Create a comparison chart for different methods"""
    if len(selected_methods) <= 1:
        return
    
    st.markdown("### Method Comparison")
    
    # Prepare data for plotting
    comparison_data = []
    for method in selected_methods:
        if method in results:
            for rank, (image_name, score) in enumerate(results[method]):
                comparison_data.append({
                    'Method': method.title(),
                    'Rank': rank + 1,
                    'Image': image_name,
                    'Score': score
                })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Create side-by-side plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Images by Method")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # For distance metrics, we want to show inverse (lower is better)
            df_plot = df.copy()
            for method in ['Euclidean', 'Manhattan']:
                if method in df_plot['Method'].values:
                    # Invert distance scores for better visualization
                    mask = df_plot['Method'] == method
                    df_plot.loc[mask, 'Score'] = 1 / (1 + df_plot.loc[mask, 'Score'])
            
            sns.barplot(data=df_plot, x='Rank', y='Score', hue='Method', ax=ax)
            ax.set_title('Similarity Scores by Rank and Method')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Similarity Score (Higher is Better)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Top 3 Images Comparison")
            pivot_df = df.pivot(index='Image', columns='Method', values='Score')
            st.dataframe(pivot_df, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌸 CBIR Flower Image Retrieval System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This Content-Based Image Retrieval (CBIR) system allows you to upload a flower image 
    and find the most similar images from our **Kaggle flower dataset** (102 categories, 6000+ images) 
    using different similarity methods.
    """)
    
    # Add dataset info banner
    st.info("🎯 **New Dataset**: Now using the comprehensive Kaggle flower dataset with 102 different flower categories!")
    
    # Load CBIR system
    cbir_system = load_cbir_system()
    feature_extractor = load_feature_extractor()
    
    if cbir_system is None or feature_extractor is None:
        st.stop()
    
    # Sidebar for method selection
    st.sidebar.markdown("## Similarity Methods")
    st.sidebar.markdown("Select which similarity methods to use:")
    
    # Method selection toggles
    use_cosine = st.sidebar.checkbox("Cosine Similarity", value=True, 
                                   help="Measures the cosine of the angle between feature vectors")
    use_euclidean = st.sidebar.checkbox("Euclidean Distance", value=True,
                                      help="Measures the straight-line distance between feature vectors")
    use_manhattan = st.sidebar.checkbox("Manhattan Distance", value=True,
                                      help="Measures the sum of absolute differences between feature vectors")
    
    # Collect selected methods
    selected_methods = []
    if use_cosine:
        selected_methods.append('cosine')
    if use_euclidean:
        selected_methods.append('euclidean')
    if use_manhattan:
        selected_methods.append('manhattan')
    
    if not selected_methods:
        st.warning("Please select at least one similarity method.")
        st.stop()
    
    # Number of results
    top_k = st.sidebar.slider("Number of results to show:", min_value=1, max_value=5, value=3)
    
    # Dataset information
    st.sidebar.markdown("## Dataset Information")
    dataset_images = cbir_system.get_database_image_names()
    st.sidebar.markdown(f"**Total images in database:** {len(dataset_images)}")
    
    # Count categories
    dataset_path = Path("dataset")
    if dataset_path.exists():
        categories = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        st.sidebar.markdown(f"**Flower categories:** {len(categories)}")
    
    st.sidebar.markdown("**Available methods:** Cosine Similarity, Euclidean Distance, Manhattan Distance")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload Query Image")
        uploaded_file = st.file_uploader(
            "Choose a flower image...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of a flower to find similar images in the database"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Query Image", use_column_width=True)
            
            # Process image and extract features
            with st.spinner("Processing image and extracting features..."):
                image_array, temp_path = process_uploaded_image(uploaded_file)
                
                if image_array is not None and temp_path is not None:
                    try:
                        # Extract features from uploaded image
                        features = feature_extractor.extract_all_features(temp_path)
                        query_features = features['combined']
                        
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        st.success("Features extracted successfully!")
                        
                        # Search button
                        if st.button("🔍 Search Similar Images", type="primary"):
                            with st.spinner("Searching for similar images..."):
                                # Query the database
                                results = cbir_system.query_image(
                                    query_features, 
                                    selected_methods=selected_methods, 
                                    top_k=top_k
                                )
                                
                                # Store results in session state
                                st.session_state['search_results'] = results
                                st.session_state['selected_methods'] = selected_methods
                    
                    except Exception as e:
                        st.error(f"Error extracting features: {e}")
    
    with col2:
        st.markdown("### Search Results")
        
        # Display results if available
        if 'search_results' in st.session_state and st.session_state['search_results']:
            results = st.session_state['search_results']
            methods = st.session_state.get('selected_methods', selected_methods)
            
            # Create tabs for each method
            if len(methods) > 1:
                tabs = st.tabs([f"{method.title()}" for method in methods])
                
                for i, method in enumerate(methods):
                    with tabs[i]:
                        if method in results:
                            display_similarity_results(results[method], method, "dataset")
                        else:
                            st.warning(f"No results available for {method}")
            else:
                # Single method display
                method = methods[0]
                if method in results:
                    display_similarity_results(results[method], method, "dataset")
            
            # Comparison chart
            if len(methods) > 1:
                with st.expander("📊 Method Comparison Analysis", expanded=False):
                    create_comparison_chart(results, methods)
        
        else:
            st.info("Upload an image and click 'Search Similar Images' to see results.")
    
    # Dataset preview
    with st.expander("📂 Kaggle Flower Dataset Preview", expanded=False):
        st.markdown("### Sample Images from Different Categories")
        
        dataset_path = Path("dataset")
        if dataset_path.exists():
            # Get all category directories
            categories = [d for d in dataset_path.iterdir() if d.is_dir()]
            categories = sorted(categories, key=lambda x: int(x.name))[:12]  # Show first 12 categories
            
            if categories:
                st.markdown(f"**Showing sample images from {len(categories)} categories (out of 102 total)**")
                
                # Create a grid layout
                cols = st.columns(4)
                
                for i, category_dir in enumerate(categories):
                    col = cols[i % 4]
                    
                    # Get first image from this category
                    images = list(category_dir.glob("*.jpg"))
                    if images:
                        with col:
                            try:
                                image = Image.open(images[0])
                                # Resize for display
                                image = image.resize((150, 150))
                                st.image(image, caption=f"Category {category_dir.name}", width=150)
                            except Exception as e:
                                st.write(f"Category {category_dir.name}: Error loading image")
                    
                    # Show category stats
                    if i < 8:  # Show stats for first 8 categories
                        with col:
                            st.write(f"**{len(images)} images**")
                
                # Dataset statistics
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_categories = len([d for d in dataset_path.iterdir() if d.is_dir()])
                    st.metric("Total Categories", total_categories)
                
                with col2:
                    # Count total images (this might be slow for large datasets)
                    if 'total_image_count' not in st.session_state:
                        total_images = sum(len(list(cat.glob("*.jpg"))) for cat in dataset_path.iterdir() if cat.is_dir())
                        st.session_state['total_image_count'] = total_images
                    st.metric("Total Images", st.session_state['total_image_count'])
                
                with col3:
                    avg_per_category = st.session_state.get('total_image_count', 0) // total_categories if total_categories > 0 else 0
                    st.metric("Avg Images/Category", avg_per_category)
            
            else:
                st.warning("No category directories found in dataset.")
        else:
            st.error("Dataset directory not found. Please download the dataset first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>CBIR System using Multiple Similarity Methods</p>
    <p>Methods: Cosine Similarity, Euclidean Distance, Manhattan Distance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()