"""
Feature extraction module for CBIR system
Extracts color histograms, texture features, and other visual features from images
"""
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

class FeatureExtractor:
    """Extract visual features from images for content-based image retrieval"""
    
    def __init__(self):
        self.feature_methods = {
            'color_histogram': self.extract_color_histogram,
            'hsv_histogram': self.extract_hsv_histogram,
            'texture_lbp': self.extract_texture_lbp,
            'color_moments': self.extract_color_moments
        }
    
    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        """
        Extract RGB color histogram features
        
        Args:
            image: numpy array of the image (H, W, 3)
            bins: tuple of number of bins for each channel
            
        Returns:
            numpy array: flattened histogram features
        """
        # Calculate histogram for each channel
        hist_r = cv2.calcHist([image], [0], None, [bins[0]], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins[1]], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [bins[2]], [0, 256])
        
        # Normalize histograms
        hist_r = hist_r.flatten() / np.sum(hist_r)
        hist_g = hist_g.flatten() / np.sum(hist_g)
        hist_b = hist_b.flatten() / np.sum(hist_b)
        
        # Concatenate all histograms
        color_hist = np.concatenate([hist_r, hist_g, hist_b])
        return color_hist
    
    def extract_hsv_histogram(self, image, bins=(8, 8, 8)):
        """
        Extract HSV color histogram features
        
        Args:
            image: numpy array of the image (H, W, 3) in RGB format
            bins: tuple of number of bins for each channel
            
        Returns:
            numpy array: flattened HSV histogram features
        """
        # Convert RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv_image], [0], None, [bins[0]], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [bins[1]], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [bins[2]], [0, 256])
        
        # Normalize histograms
        hist_h = hist_h.flatten() / np.sum(hist_h)
        hist_s = hist_s.flatten() / np.sum(hist_s)
        hist_v = hist_v.flatten() / np.sum(hist_v)
        
        # Concatenate all histograms
        hsv_hist = np.concatenate([hist_h, hist_s, hist_v])
        return hsv_hist
    
    def extract_texture_lbp(self, image, radius=1, n_points=8):
        """
        Extract Local Binary Pattern (LBP) texture features
        
        Args:
            image: numpy array of the image
            radius: radius of the LBP operator
            n_points: number of points in the LBP operator
            
        Returns:
            numpy array: LBP histogram features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple LBP implementation
        height, width = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = gray[i, j]
                code = 0
                
                # Sample points around the center
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if x < height and y < width and gray[x, y] >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=2**n_points, range=(0, 2**n_points))
        hist = hist.astype(float)
        hist = hist / np.sum(hist)  # Normalize
        
        return hist
    
    def extract_color_moments(self, image):
        """
        Extract color moments (mean, standard deviation, skewness)
        
        Args:
            image: numpy array of the image (H, W, 3)
            
        Returns:
            numpy array: color moments features
        """
        moments = []
        
        for channel in range(3):  # RGB channels
            ch = image[:, :, channel].flatten()
            
            # First moment (mean)
            mean = np.mean(ch)
            
            # Second moment (standard deviation)
            std = np.std(ch)
            
            # Third moment (skewness)
            skewness = np.mean(((ch - mean) / std) ** 3) if std > 0 else 0
            
            moments.extend([mean, std, skewness])
        
        return np.array(moments)
    
    def extract_all_features(self, image_path):
        """
        Extract all features from an image
        
        Args:
            image_path: path to the image file
            
        Returns:
            dict: dictionary containing all extracted features
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image for consistent processing
        image = cv2.resize(image, (256, 256))
        
        features = {}
        
        # Extract different types of features
        features['color_histogram'] = self.extract_color_histogram(image)
        features['hsv_histogram'] = self.extract_hsv_histogram(image)
        features['texture_lbp'] = self.extract_texture_lbp(image)
        features['color_moments'] = self.extract_color_moments(image)
        
        # Combine all features into a single vector
        combined_features = np.concatenate([
            features['color_histogram'],
            features['hsv_histogram'], 
            features['texture_lbp'],
            features['color_moments']
        ])
        
        features['combined'] = combined_features
        
        return features

def extract_features_from_dataset(dataset_path, output_path):
    """
    Extract features from all images in the dataset
    
    Args:
        dataset_path: path to the dataset directory
        output_path: path to save the extracted features
    """
    extractor = FeatureExtractor()
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    features_db = {}
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset_path.glob(f'*{ext}'))
        image_files.extend(dataset_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images in dataset")
    
    for image_path in image_files:
        try:
            print(f"Processing {image_path.name}...")
            features = extractor.extract_all_features(image_path)
            features_db[image_path.name] = features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save features database
    np.save(output_path / 'features_db.npy', features_db)
    print(f"Features database saved to {output_path / 'features_db.npy'}")
    print(f"Processed {len(features_db)} images")
    
    return features_db

if __name__ == "__main__":
    # Extract features from the dataset
    dataset_path = "dataset"
    output_path = "features" 
    
    features_db = extract_features_from_dataset(dataset_path, output_path)
    print("Feature extraction completed!")