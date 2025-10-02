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
from visualizer import VectorVisualizer, create_visualization_summary

# Flower class names mapping
FLOWER_CLASSES = {
    "1": "pink primrose",
    "2": "hard-leaved pocket orchid",
    "3": "canterbury bells",
    "4": "sweet pea",
    "5": "english marigold",
    "6": "tiger lily",
    "7": "moon orchid",
    "8": "bird of paradise",
    "9": "monkshood",
    "10": "globe thistle",
    "11": "snapdragon",
    "12": "colt's foot",
    "13": "king protea",
    "14": "spear thistle",
    "15": "yellow iris",
    "16": "globe-flower",
    "17": "purple coneflower",
    "18": "peruvian lily",
    "19": "balloon flower",
    "20": "giant white arum lily",
    "21": "fire lily",
    "22": "pincushion flower",
    "23": "fritillary",
    "24": "red ginger",
    "25": "grape hyacinth",
    "26": "corn poppy",
    "27": "prince of wales feathers",
    "28": "stemless gentian",
    "29": "artichoke",
    "30": "sweet william",
    "31": "carnation",
    "32": "garden phlox",
    "33": "love in the mist",
    "34": "mexican aster",
    "35": "alpine sea holly",
    "36": "ruby-lipped cattleya",
    "37": "cape flower",
    "38": "great masterwort",
    "39": "siam tulip",
    "40": "lenten rose",
    "41": "barbeton daisy",
    "42": "daffodil",
    "43": "sword lily",
    "44": "poinsettia",
    "45": "bolero deep blue",
    "46": "wallflower",
    "47": "marigold",
    "48": "buttercup",
    "49": "oxeye daisy",
    "50": "common dandelion",
    "51": "petunia",
    "52": "wild pansy",
    "53": "primula",
    "54": "sunflower",
    "55": "pelargonium",
    "56": "bishop of llandaff",
    "57": "gaura",
    "58": "geranium",
    "59": "orange dahlia",
    "60": "pink-yellow dahlia",
    "61": "cautleya spicata",
    "62": "japanese anemone",
    "63": "black-eyed susan",
    "64": "silverbush",
    "65": "californian poppy",
    "66": "osteospermum",
    "67": "spring crocus",
    "68": "bearded iris",
    "69": "windflower",
    "70": "tree poppy",
    "71": "gazania",
    "72": "azalea",
    "73": "water lily",
    "74": "rose",
    "75": "thorn apple",
    "76": "morning glory",
    "77": "passion flower",
    "78": "lotus lotus",
    "79": "toad lily",
    "80": "anthurium",
    "81": "frangipani",
    "82": "clematis",
    "83": "hibiscus",
    "84": "columbine",
    "85": "desert-rose",
    "86": "tree mallow",
    "87": "magnolia",
    "88": "cyclamen",
    "89": "watercress",
    "90": "canna lily",
    "91": "hippeastrum",
    "92": "bee balm",
    "93": "ball moss",
    "94": "foxglove",
    "95": "bougainvillea",
    "96": "camellia",
    "97": "mallow",
    "98": "mexican petunia",
    "99": "bromelia",
    "100": "blanket flower",
    "101": "trumpet creeper",
    "102": "blackberry lily"
}

def get_flower_name(category_id):
    """Get flower name from category ID, with fallback to category number."""
    try:
        # Ensure category_id is a string
        category_str = str(category_id)
        return FLOWER_CLASSES.get(category_str, f"Category {category_str}")
    except:
        return f"Category {category_id}"

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
                        flower_name = get_flower_name(category)
                        image_file = parts[1]
                        st.markdown(f"**Category:** {flower_name}")
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
            st.dataframe(pivot_df, width='stretch')

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
            st.image(image, caption="Query Image", width='stretch')
            
            # Process image and extract features
            with st.spinner("Processing image and extracting features..."):
                image_array, temp_path = process_uploaded_image(uploaded_file)
                
                if image_array is not None and temp_path is not None:
                    try:
                        # Extract features from uploaded image
                        features = feature_extractor.extract_all_features(temp_path)
                        query_features = features['combined']
                        
                        # Store query features in session state
                        st.session_state['query_features'] = query_features
                        
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
            
            # Vector Visualization Section
            if 'query_features' in locals() or 'query_features' in st.session_state:
                st.markdown("---")
                st.markdown("### 🎯 Vector Analysis & Similarity Visualization")
                
                # Store query features in session state if not already there
                if 'query_features' in locals() and 'query_features' not in st.session_state:
                    st.session_state['query_features'] = query_features
                
                current_query_features = st.session_state.get('query_features')
                
                if current_query_features is not None:
                    # Initialize visualizer
                    visualizer = VectorVisualizer(feature_extractor)
                    
                    # Prepare data for visualization
                    method_results = {}
                    for method in methods:
                        if method in results:
                            # Extract features and distances from results
                            result_data = results[method]
                            features_list = []
                            distances_list = []
                            names_list = []
                            
                            for image_name, distance in result_data[:top_k]:
                                # Get features for this image
                                image_path = Path("dataset") / image_name
                                if image_path.exists():
                                    try:
                                        img_features = feature_extractor.extract_all_features(str(image_path))
                                        features_list.append(img_features['combined'])
                                        distances_list.append(distance)
                                        
                                        # Get flower name
                                        parts = Path(image_name).parts
                                        if len(parts) >= 2:
                                            category = parts[0]
                                            flower_name = get_flower_name(category)
                                            names_list.append(flower_name)
                                        else:
                                            names_list.append(image_name)
                                    except Exception as e:
                                        continue
                            
                            if features_list:
                                method_results[method] = (features_list, distances_list, names_list)
                    
                    # Create visualization tabs
                    viz_tabs = st.tabs(["📈 Vector Space", "📊 Distance Analysis", "🔬 Multi-Method Comparison", "🔥 Feature Heatmap"])
                    
                    with viz_tabs[0]:
                        st.markdown("#### Feature Vector Visualization in 2D Space")
                        st.info("This plot shows how images are positioned in feature space using dimensionality reduction (PCA). Closer points indicate more similar images.")
                        
                        # Reduction method selector
                        reduction_method = st.selectbox(
                            "Dimensionality Reduction Method:",
                            ["PCA", "tsne"],
                            help="PCA preserves global structure, t-SNE better for local neighborhoods"
                        )
                        
                        # Create vector plots for each method
                        for method in methods:
                            if method in method_results:
                                features, distances, names = method_results[method]
                                
                                st.markdown(f"**{method.title()} Method**")
                                fig = visualizer.create_vector_scatter_plot(
                                    current_query_features, features, 
                                    retrieved_names=names, method_name=method,
                                    reduction_method=reduction_method.lower()
                                )
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                    
                    with viz_tabs[1]:
                        st.markdown("#### Distance and Similarity Analysis")
                        st.info("Detailed analysis of how similarity scores relate to feature vectors and geometric relationships.")
                        
                        for method in methods:
                            if method in method_results:
                                features, distances, names = method_results[method]
                                
                                st.markdown(f"**{method.title()} Method Analysis**")
                                fig = visualizer.create_distance_visualization(
                                    current_query_features, features, distances, method, names
                                )
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                                
                                # Add interpretation
                                if method.lower() == 'cosine':
                                    st.markdown("""
                                    **Cosine Similarity Interpretation:**
                                    - Values closer to 1.0 indicate higher similarity
                                    - Measures the angle between feature vectors
                                    - Insensitive to vector magnitude
                                    """)
                                elif method.lower() == 'euclidean':
                                    st.markdown("""
                                    **Euclidean Distance Interpretation:**
                                    - Lower values indicate higher similarity
                                    - Measures straight-line distance in feature space
                                    - Sensitive to all feature dimensions equally
                                    """)
                                elif method.lower() == 'manhattan':
                                    st.markdown("""
                                    **Manhattan Distance Interpretation:**
                                    - Lower values indicate higher similarity
                                    - Sum of absolute differences in each dimension
                                    - More robust to outliers than Euclidean
                                    """)
                    
                    with viz_tabs[2]:
                        if len(methods) > 1:
                            st.markdown("#### Multi-Method Comparison Dashboard")
                            st.info("Compare how different similarity methods rank and score the same images.")
                            
                            fig = visualizer.create_multi_method_comparison(
                                current_query_features, method_results, methods
                            )
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                            
                            # Method comparison insights
                            st.markdown("#### 🔍 Method Insights")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Cosine Similarity**")
                                st.markdown("- Best for: Style and texture matching")
                                st.markdown("- Focus: Shape of feature vector")
                                st.markdown("- Robust to: Lighting variations")
                            
                            with col2:
                                st.markdown("**Euclidean Distance**")
                                st.markdown("- Best for: Overall feature similarity")
                                st.markdown("- Focus: Magnitude and direction")
                                st.markdown("- Sensitive to: All feature changes")
                            
                            with col3:
                                st.markdown("**Manhattan Distance**")
                                st.markdown("- Best for: Color-based matching")
                                st.markdown("- Focus: Individual feature differences")
                                st.markdown("- Robust to: Outlier features")
                        else:
                            st.info("Select multiple similarity methods to see comparison analysis.")
                    
                    with viz_tabs[3]:
                        st.markdown("#### Feature Vector Heatmap")
                        st.info("Raw feature values comparison between query and retrieved images.")
                        
                        # Use primary method for heatmap
                        primary_method = methods[0]
                        if primary_method in method_results:
                            features, _, names = method_results[primary_method]
                            
                            fig = visualizer.create_feature_heatmap(
                                current_query_features, features, names
                            )
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                            
                            st.markdown("""
                            **Feature Heatmap Interpretation:**
                            - Each row represents an image (query + retrieved)
                            - Each column represents a feature dimension
                            - Color intensity shows feature values
                            - Similar patterns indicate similar images
                            """)
                else:
                    st.warning("Query features not available for visualization. Please search for images first.")
        
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
                                flower_name = get_flower_name(category_dir.name)
                                st.image(image, caption=flower_name, width=150)
                            except Exception as e:
                                flower_name = get_flower_name(category_dir.name)
                                st.write(f"{flower_name}: Error loading image")
                    
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