# CBIR (Content-Based Image Retrieval) Mini System 🌸

A comprehensive Content-Based Image Retrieval system that allows users to upload flower images and retrieve the top 3 most similar images from a database using multiple similarity methods.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/AhmedelBamby/Computer-Vision-AAST-2025?quickstart=1)
## 🚀 Features

- **Image Upload**: Upload flower images through an intuitive web interface
- **Multiple Similarity Methods**: 
  - Cosine Similarity
  - Euclidean Distance  
  - Manhattan Distance
- **Method Toggle**: Enable/disable specific similarity methods
- **Top-K Results**: Retrieve top 3 most similar images
- **Feature Extraction**: Advanced feature extraction including:
  - RGB Color Histograms
  - HSV Color Histograms
  - Local Binary Pattern (LBP) Texture Features
  - Color Moments (mean, std, skewness)
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Comparison Analysis**: Visual comparison between different methods
- **Dataset Preview**: Browse the flower dataset

## 🏗️ Project Structure

```
cbir_system/
├── app.py                      # Main Streamlit web application
├── feature_extractor.py        # Feature extraction module
├── similarity_calculator.py    # Similarity calculation module  
├── create_sample_dataset.py    # Dataset creation script
├── download_dataset.py         # Alternative dataset download script
├── dataset/                    # Flower images dataset
│   ├── red_flower_1.jpg
│   ├── blue_flower_1.jpg
│   ├── yellow_flower_1.jpg
│   └── ...
└── features/                   # Extracted features database
    └── features_db.npy
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/AhmedelBamby/Computer-Vision-AAST-2025.git
cd Computer-Vision-AAST-2025/cbir_system
```

### Step 2: Install Dependencies

```bash
pip install streamlit opencv-python-headless pillow numpy matplotlib scikit-learn seaborn pandas
```

### Step 3: Create Sample Dataset

```bash
python create_sample_dataset.py
```

This will create 16 synthetic flower images with different colors (red, blue, yellow, purple, orange, pink).

### Step 4: Extract Features

```bash
python feature_extractor.py
```

This processes all images and creates a features database for fast retrieval.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📊 How It Works

### 1. Feature Extraction

The system extracts multiple types of features from each image:

- **RGB Color Histogram**: Captures color distribution in RGB space
- **HSV Color Histogram**: Captures color information in HSV space (more perceptually uniform)
- **LBP Texture Features**: Local Binary Pattern for texture analysis
- **Color Moments**: Statistical measures (mean, standard deviation, skewness) for each color channel

### 2. Similarity Calculation Methods

#### Cosine Similarity
- Measures the cosine of the angle between feature vectors
- Values range from -1 to 1 (higher is more similar)
- Good for high-dimensional data
- Formula: `cos(θ) = (A · B) / (||A|| ||B||)`

#### Euclidean Distance  
- Measures straight-line distance between feature vectors
- Lower values indicate higher similarity
- Sensitive to magnitude differences
- Formula: `d = √(Σ(ai - bi)²)`

#### Manhattan Distance
- Sum of absolute differences between feature vectors  
- Lower values indicate higher similarity
- Less sensitive to outliers than Euclidean
- Formula: `d = Σ|ai - bi|`

### 3. Retrieval Process

1. User uploads a query image
2. System extracts features from the query image
3. Features are compared against the database using selected methods
4. Top-K most similar images are returned and displayed
5. Results are presented with similarity scores

## 🎯 Usage Guide

### Web Interface

1. **Upload Image**: Click "Browse files" and select a flower image
2. **Select Methods**: Use the sidebar toggles to enable/disable similarity methods:
   - ✅ Cosine Similarity
   - ✅ Euclidean Distance  
   - ✅ Manhattan Distance
3. **Set Results Count**: Use the slider to choose how many results to show (1-5)
4. **Search**: Click "🔍 Search Similar Images" to find similar images
5. **View Results**: Results are displayed in tabs for each method
6. **Compare Methods**: Expand "Method Comparison Analysis" for visual comparison

### Features

- **Real-time Processing**: Images are processed and searched in real-time
- **Method Comparison**: Side-by-side comparison of different similarity methods
- **Dataset Preview**: Browse sample images from the dataset
- **Responsive Design**: Works on desktop and mobile browsers

## 🔧 Technical Details

### Feature Vector Composition

Each image is represented by a combined feature vector containing:
- RGB histogram features (24 dimensions: 8 bins × 3 channels)
- HSV histogram features (24 dimensions: 8 bins × 3 channels)  
- LBP texture features (256 dimensions for 8-point LBP)
- Color moments (9 dimensions: 3 moments × 3 channels)

**Total**: ~313 dimensional feature vector per image

### Performance Optimizations

- **Caching**: Streamlit caching for faster feature database loading
- **Vectorized Operations**: NumPy for efficient computation
- **Preprocessing**: Images resized to 256×256 for consistent processing
- **Feature Normalization**: Zero-mean, unit-variance normalization

### Dataset

The system includes a synthetic flower dataset with:
- 16 flower images in 6 different colors
- Multiple variations per color for better testing
- 256×256 pixel resolution
- JPEG format for web compatibility

## 📈 Results & Analysis

### Method Comparison

Different similarity methods excel in different scenarios:

- **Cosine Similarity**: Best for color-based similarity, robust to illumination changes
- **Euclidean Distance**: Good overall performance, sensitive to feature magnitude
- **Manhattan Distance**: More robust to outliers, good for noisy images

### Example Results

When querying with a red flower image:
1. **Cosine Similarity**: Focuses on color distribution patterns
2. **Euclidean Distance**: Considers overall feature similarity
3. **Manhattan Distance**: Emphasizes dominant features

## 🚨 Troubleshooting

### Common Issues

1. **ImportError: libGL.so.1**: Install system dependencies:
   ```bash
   sudo apt-get install libgl1-mesa-dri libglib2.0-0
   ```

2. **OpenCV Display Issues**: Use opencv-python-headless:
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

3. **Memory Issues**: Reduce image resolution in feature_extractor.py

4. **Port Already in Use**: Change Streamlit port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Feature Database Issues

If features database is corrupted or missing:
```bash
rm features/features_db.npy
python feature_extractor.py
```

## 🔮 Future Enhancements

- [ ] Deep learning features (CNN-based)
- [ ] Support for additional image formats
- [ ] Batch upload and processing
- [ ] Advanced filtering options
- [ ] Real-world flower dataset integration
- [ ] Performance metrics and evaluation
- [ ] User feedback and rating system
- [ ] Export functionality for results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ahmed ElBamby**
- GitHub: [@AhmedelBamby](https://github.com/AhmedelBamby)
- University: AAST (Arab Academy for Science, Technology & Maritime Transport)
- Course: Computer Vision 2025

## 📚 References

- Smeulders, A. W. M., et al. (2000). Content-based image retrieval at the end of the early years. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Liu, Y., Zhang, D., Lu, G., & Ma, W. Y. (2007). A survey of content-based image retrieval with high-level semantics.
- OpenCV Documentation: https://docs.opencv.org/
- Streamlit Documentation: https://docs.streamlit.io/

---

**Built with ❤️ for Computer Vision course at AAST 2025**