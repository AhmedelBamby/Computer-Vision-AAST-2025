# API Documentation - CBIR System

## 📚 Module Reference

### Core Classes and Functions

---

## 🔧 feature_extractor.py

### Class: `FeatureExtractor`

Primary class for extracting visual features from images.

#### **Constructor**
```python
def __init__(self):
    """
    Initialize the FeatureExtractor with LBP descriptor and multiprocessing pool.
    
    Attributes:
        lbp_descriptor: Local Binary Pattern descriptor (radius=1, neighbors=8)
        pool: Multiprocessing pool for parallel feature extraction
    """
```

#### **Methods**

##### `extract_rgb_histogram(image_path: str) -> np.ndarray`
```python
def extract_rgb_histogram(self, image_path):
    """
    Extract RGB color histogram from image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Flattened RGB histogram (512 values compressed to 64)
        
    Raises:
        IOError: If image cannot be loaded
        ValueError: If image format is unsupported
    """
```

##### `extract_hsv_histogram(image_path: str) -> np.ndarray`
```python
def extract_hsv_histogram(self, image_path):
    """
    Extract HSV color histogram from image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Flattened HSV histogram (512 values compressed to 64)
        
    Notes:
        HSV provides better perceptual color representation than RGB
    """
```

##### `extract_lbp_histogram(image_path: str) -> np.ndarray`
```python
def extract_lbp_histogram(self, image_path):
    """
    Extract Local Binary Pattern texture features.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: LBP histogram (59 uniform patterns)
        
    Details:
        - Radius: 1 pixel
        - Neighbors: 8 sampling points
        - Method: 'uniform' patterns only
    """
```

##### `extract_color_moments(image_path: str) -> np.ndarray`
```python
def extract_color_moments(self, image_path):
    """
    Extract statistical color moments from image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Color moments (9 values)
        
    Features:
        - Mean (3 channels)
        - Standard deviation (3 channels)  
        - Skewness (3 channels)
    """
```

##### `extract_all_features(image_path: str) -> dict`
```python
def extract_all_features(self, image_path):
    """
    Extract all feature types and combine into single vector.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: {
            'rgb_hist': np.ndarray,
            'hsv_hist': np.ndarray,
            'lbp_hist': np.ndarray,
            'color_moments': np.ndarray,
            'combined': np.ndarray  # 313-dimensional vector
        }
        
    Performance:
        - Time: ~0.15 seconds per image
        - Memory: ~5MB during processing
    """
```

##### `extract_features_batch(image_paths: List[str]) -> dict`
```python
def extract_features_batch(self, image_paths):
    """
    Extract features from multiple images using multiprocessing.
    
    Args:
        image_paths (List[str]): List of image file paths
        
    Returns:
        dict: {image_path: feature_vector} mapping
        
    Performance:
        - Multiprocessing: 8 workers (configurable)
        - Speedup: ~6x faster than sequential processing
    """
```

---

## 🔍 similarity_calculator.py

### Class: `CBIRSystem`

Main system for content-based image retrieval and similarity calculation.

#### **Constructor**
```python
def __init__(self, features_db_path: str):
    """
    Initialize CBIR system with pre-computed feature database.
    
    Args:
        features_db_path (str): Path to .npy feature database file
        
    Attributes:
        features_db (dict): Loaded feature database
        image_paths (list): List of all image paths in database
        
    Raises:
        FileNotFoundError: If feature database doesn't exist
        ValueError: If database format is invalid
    """
```

#### **Methods**

##### `cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float`
```python
def cosine_similarity(self, vec1, vec2):
    """
    Calculate cosine similarity between two feature vectors.
    
    Args:
        vec1 (np.ndarray): First feature vector
        vec2 (np.ndarray): Second feature vector
        
    Returns:
        float: Cosine similarity [-1, 1] (1 = identical)
        
    Formula:
        cos(θ) = (A·B) / (||A|| × ||B||)
    """
```

##### `euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float`
```python
def euclidean_distance(self, vec1, vec2):
    """
    Calculate Euclidean distance between two feature vectors.
    
    Args:
        vec1 (np.ndarray): First feature vector
        vec2 (np.ndarray): Second feature vector
        
    Returns:
        float: Euclidean distance [0, ∞] (0 = identical)
        
    Formula:
        d = √(Σ(ai - bi)²)
    """
```

##### `manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float`
```python
def manhattan_distance(self, vec1, vec2):
    """
    Calculate Manhattan distance between two feature vectors.
    
    Args:
        vec1 (np.ndarray): First feature vector
        vec2 (np.ndarray): Second feature vector
        
    Returns:
        float: Manhattan distance [0, ∞] (0 = identical)
        
    Formula:
        d = Σ|ai - bi|
    """
```

##### `query_image(query_features: np.ndarray, selected_methods: List[str], top_k: int) -> dict`
```python
def query_image(self, query_features, selected_methods=['cosine'], top_k=5):
    """
    Query the database for similar images using specified methods.
    
    Args:
        query_features (np.ndarray): 313-dimensional query feature vector
        selected_methods (List[str]): ['cosine', 'euclidean', 'manhattan']
        top_k (int): Number of top results to return
        
    Returns:
        dict: {
            'cosine': [(image_path, similarity_score), ...],
            'euclidean': [(image_path, distance), ...],
            'manhattan': [(image_path, distance), ...]
        }
        
    Performance:
        - Time: ~2.4 seconds for 6,552 images
        - Memory: ~8MB database + query overhead
    """
```

---

## 📊 visualizer.py

### Class: `VectorVisualizer`

Advanced visualization system for understanding similarity relationships.

#### **Constructor**
```python
def __init__(self, feature_extractor=None):
    """
    Initialize vector visualization system.
    
    Args:
        feature_extractor (FeatureExtractor, optional): Feature extractor instance
        
    Attributes:
        colors (dict): Color scheme for different methods
        feature_extractor: Optional feature extractor for on-demand processing
    """
```

#### **Methods**

##### `reduce_dimensions(features: np.ndarray, method: str, n_components: int) -> np.ndarray`
```python
def reduce_dimensions(self, features, method='pca', n_components=2):
    """
    Reduce feature vector dimensions for visualization.
    
    Args:
        features (np.ndarray): High-dimensional feature array
        method (str): 'pca' or 'tsne'
        n_components (int): Output dimensions (typically 2)
        
    Returns:
        np.ndarray: Reduced dimensional representation
        
    Methods:
        - PCA: Linear, preserves global structure
        - t-SNE: Non-linear, preserves local neighborhoods
    """
```

##### `create_vector_scatter_plot(...) -> plotly.graph_objects.Figure`
```python
def create_vector_scatter_plot(self, query_features, retrieved_features, 
                             query_name="Query Image", retrieved_names=None,
                             method_name="Similarity", reduction_method='pca'):
    """
    Create interactive 2D scatter plot of feature vectors.
    
    Args:
        query_features (np.ndarray): Query image feature vector
        retrieved_features (List[np.ndarray]): Retrieved image feature vectors
        query_name (str): Label for query point
        retrieved_names (List[str]): Labels for retrieved points
        method_name (str): Similarity method name
        reduction_method (str): 'pca' or 'tsne'
        
    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot
        
    Features:
        - Hover tooltips with image information
        - Query highlighted as star marker
        - Retrieved images numbered by rank
        - Zoom, pan, and selection capabilities
    """
```

##### `create_distance_visualization(...) -> plotly.graph_objects.Figure`
```python
def create_distance_visualization(self, query_features, retrieved_features, 
                                distances, method_name, retrieved_names=None):
    """
    Create comprehensive distance analysis dashboard.
    
    Args:
        query_features (np.ndarray): Query image feature vector
        retrieved_features (List[np.ndarray]): Retrieved image feature vectors
        distances (List[float]): Similarity/distance scores
        method_name (str): Method name ('cosine', 'euclidean', 'manhattan')
        retrieved_names (List[str]): Image names for labeling
        
    Returns:
        plotly.graph_objects.Figure: Multi-panel dashboard
        
    Components:
        - Bar chart of similarity scores
        - Vector magnitude comparison
        - Angle visualization (cosine method)
        - Distance matrix heatmap
    """
```

##### `create_multi_method_comparison(...) -> plotly.graph_objects.Figure`
```python
def create_multi_method_comparison(self, query_features, retrieved_results, method_names):
    """
    Create side-by-side comparison of multiple similarity methods.
    
    Args:
        query_features (np.ndarray): Query image feature vector
        retrieved_results (dict): Results from multiple methods
        method_names (List[str]): List of method names to compare
        
    Returns:
        plotly.graph_objects.Figure: Comparison dashboard
        
    Visualizations:
        - Score comparison for top-K results
        - Vector space overlay with method colors
        - Ranking comparison across methods
        - Score distribution analysis
    """
```

##### `create_feature_heatmap(...) -> plotly.graph_objects.Figure`
```python
def create_feature_heatmap(self, query_features, retrieved_features, retrieved_names=None):
    """
    Create heatmap visualization of raw feature values.
    
    Args:
        query_features (np.ndarray): Query image feature vector
        retrieved_features (List[np.ndarray]): Retrieved image feature vectors
        retrieved_names (List[str]): Image names for row labels
        
    Returns:
        plotly.graph_objects.Figure: Feature heatmap
        
    Features:
        - Color-coded feature intensity
        - Hover tooltips with exact values
        - Pattern recognition assistance
        - Comparative analysis across images
    """
```

---

## 🌸 app.py - Streamlit Interface

### Helper Functions

##### `get_flower_name(category_id: str) -> str`
```python
def get_flower_name(category_id):
    """
    Convert numeric category ID to flower species name.
    
    Args:
        category_id (str): Numeric category identifier
        
    Returns:
        str: Human-readable flower species name
        
    Examples:
        get_flower_name('1') → 'pink primrose'
        get_flower_name('25') → 'grape hyacinth'
        get_flower_name('102') → 'blackberry lily'
    """
```

##### `process_uploaded_image(uploaded_file) -> Tuple[np.ndarray, str]`
```python
def process_uploaded_image(uploaded_file):
    """
    Process uploaded image file for feature extraction.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple[np.ndarray, str]: (image_array, temp_file_path)
        
    Processing:
        - Validates file format
        - Creates temporary file
        - Converts to OpenCV format
        - Handles errors gracefully
    """
```

##### `display_similarity_results(results, method_name, dataset_path)`
```python
def display_similarity_results(results, method_name, dataset_path):
    """
    Display similarity search results in Streamlit interface.
    
    Args:
        results: List of (image_name, score) tuples
        method_name (str): Similarity method name
        dataset_path (str): Path to dataset directory
        
    Features:
        - Grid layout for result images
        - Similarity scores with color coding
        - Flower species names instead of IDs
        - Error handling for missing images
    """
```

### Constants

##### `FLOWER_CLASSES: dict`
```python
FLOWER_CLASSES = {
    "1": "pink primrose",
    "2": "hard-leaved pocket orchid",
    # ... 102 total species mappings
    "102": "blackberry lily"
}
```

---

## 🧪 Testing Functions

### test_system.py

##### `test_feature_extraction()`
```python
def test_feature_extraction():
    """
    Test feature extraction functionality with sample images.
    
    Tests:
        - Feature vector dimensions (313)
        - Feature value ranges
        - Processing time benchmarks
        - Error handling
    """
```

##### `test_similarity_calculation()`
```python
def test_similarity_calculation():
    """
    Test similarity calculation algorithms.
    
    Tests:
        - Cosine similarity range [-1, 1]
        - Euclidean distance non-negativity
        - Manhattan distance computation
        - Identical image similarity (should be 1.0 or 0.0)
    """
```

### test_visualizations.py

##### `test_visualizations()`
```python
def test_visualizations():
    """
    Comprehensive test of visualization module.
    
    Tests:
        - Vector scatter plot creation
        - Distance visualization generation
        - Multi-method comparison
        - Feature heatmap functionality
        - PCA and t-SNE dimensionality reduction
    """
```

---

## 📊 Usage Examples

### Basic Feature Extraction
```python
from feature_extractor import FeatureExtractor

# Initialize extractor
extractor = FeatureExtractor()

# Extract features from single image
features = extractor.extract_all_features('path/to/image.jpg')
feature_vector = features['combined']  # 313-dimensional

# Extract features from multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch_features = extractor.extract_features_batch(image_paths)
```

### Similarity Search
```python
from similarity_calculator import CBIRSystem

# Initialize CBIR system
cbir = CBIRSystem('features/features_db.npy')

# Query with multiple methods
results = cbir.query_image(
    query_features=feature_vector,
    selected_methods=['cosine', 'euclidean', 'manhattan'],
    top_k=5
)

# Access results
cosine_results = results['cosine']  # List of (image_path, similarity)
euclidean_results = results['euclidean']  # List of (image_path, distance)
```

### Vector Visualization
```python
from visualizer import VectorVisualizer

# Initialize visualizer
viz = VectorVisualizer()

# Create scatter plot
fig = viz.create_vector_scatter_plot(
    query_features=query_vector,
    retrieved_features=[feat1, feat2, feat3],
    retrieved_names=['Rose', 'Tulip', 'Daisy'],
    method_name='cosine',
    reduction_method='pca'
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
```

### Complete Workflow
```python
# 1. Extract features from query image
query_features = extractor.extract_all_features('query.jpg')['combined']

# 2. Search for similar images
results = cbir.query_image(query_features, ['cosine'], top_k=3)

# 3. Visualize results
retrieved_features = [extract_features(img) for img, _ in results['cosine']]
fig = viz.create_vector_scatter_plot(query_features, retrieved_features)

# 4. Display results
st.plotly_chart(fig)
```

---

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Custom worker count for multiprocessing
export CBIR_WORKERS=8

# Optional: Custom cache directory
export CBIR_CACHE_DIR=/path/to/cache

# Optional: Debug mode
export CBIR_DEBUG=True
```

### Streamlit Configuration
```toml
# .streamlit/config.toml
[server]
port = 8501
address = "0.0.0.0"

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

---

*This API documentation provides complete reference for all classes, methods, and functions in the CBIR system. Each component is designed for modularity, testability, and extensibility.*