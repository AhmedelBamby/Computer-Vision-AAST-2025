"""
Feature extraction module for CBIR system
Extracts color histograms, texture features, and other visual features from images
using multiprocessing (ProcessPoolExecutor)
"""
import cv2
import numpy as np
import os
from pathlib import Path
import concurrent.futures
import time
import json

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
        """Extract RGB color histogram features"""
        hist_r = cv2.calcHist([image], [0], None, [bins[0]], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins[1]], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [bins[2]], [0, 256])
        
        hist_r = hist_r.flatten() / np.sum(hist_r)
        hist_g = hist_g.flatten() / np.sum(hist_g)
        hist_b = hist_b.flatten() / np.sum(hist_b)
        
        return np.concatenate([hist_r, hist_g, hist_b])
    
    def extract_hsv_histogram(self, image, bins=(8, 8, 8)):
        """Extract HSV color histogram features"""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [bins[0]], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [bins[1]], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [bins[2]], [0, 256])
        
        hist_h = hist_h.flatten() / np.sum(hist_h)
        hist_s = hist_s.flatten() / np.sum(hist_s)
        hist_v = hist_v.flatten() / np.sum(hist_v)
        
        return np.concatenate([hist_h, hist_s, hist_v])
    
    def extract_texture_lbp(self, image, radius=1, n_points=8):
        """Extract Local Binary Pattern (LBP) texture features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        height, width = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = gray[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if x < height and y < width and gray[x, y] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        
        hist, _ = np.histogram(lbp.ravel(), bins=2**n_points, range=(0, 2**n_points))
        hist = hist.astype(float)
        return hist / np.sum(hist)
    
    def extract_color_moments(self, image):
        """Extract color moments (mean, std, skewness)"""
        moments = []
        for channel in range(3):
            ch = image[:, :, channel].flatten()
            mean = np.mean(ch)
            std = np.std(ch)
            skewness = np.mean(((ch - mean) / std) ** 3) if std > 0 else 0
            moments.extend([mean, std, skewness])
        return np.array(moments)
    
    def extract_all_features(self, image_path):
        """Extract all features from an image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        features = {
            'color_histogram': self.extract_color_histogram(image),
            'hsv_histogram': self.extract_hsv_histogram(image),
            'texture_lbp': self.extract_texture_lbp(image),
            'color_moments': self.extract_color_moments(image)
        }
        
        combined_features = np.concatenate([
            features['color_histogram'],
            features['hsv_histogram'],
            features['texture_lbp'],
            features['color_moments']
        ])
        features['combined'] = combined_features
        return features


def process_single_image(image_path, dataset_path):
    """Standalone worker function for multiprocessing"""
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(image_path)
    relative_path = image_path.relative_to(dataset_path)
    key = str(relative_path).replace("\\", "/")
    return key, features


def extract_features_from_dataset(dataset_path, output_path, max_workers=4):
    """
    Extract features from all images in the dataset using multiprocessing
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    print(f"🔍 Scanning dataset: {dataset_path}")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        Path(root) / file
        for root, _, files in os.walk(dataset_path)
        for file in files
        if Path(file).suffix.lower() in image_extensions
    ]
    
    print(f"🔍 Found {len(image_files)} images in dataset")
    if not image_files:
        print("❌ No images found in dataset!")
        return {}
    
    features_db = {}
    successful = 0
    failed = 0
    
    print(f"🚀 Starting feature extraction with {max_workers} processes...")
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, img, dataset_path): img for img in image_files}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                key, features = future.result()
                features_db[key] = features
                successful += 1
                if i % 100 == 0:
                    print(f"📊 Progress: {i}/{len(image_files)} ({(i/len(image_files))*100:.1f}%)")
                elif i % 10 == 0:
                    print(".", end="", flush=True)
            except Exception as e:
                failed += 1
                if failed <= 10:
                    print(f"\n❌ Error: {e}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save results
    features_db_path = output_path / "features_db.npy"
    np.save(features_db_path, features_db)
    
    stats = {
        "total_images": len(image_files),
        "successful": successful,
        "failed": failed,
        "processing_time_seconds": processing_time,
        "images_per_second": successful / processing_time if processing_time > 0 else 0,
        "processes_used": max_workers,
        "features_per_image": len(next(iter(features_db.values()))['combined']) if features_db else 0,
        "dataset_path": str(dataset_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    stats_path = output_path / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📊 Feature Extraction Summary:")
    print(f"✅ Successfully processed: {successful:,}")
    print(f"❌ Failed: {failed:,}")
    print(f"⏱️ Total time: {processing_time:.2f} seconds")
    print(f"🚀 Speed: {successful/processing_time:.2f} images/sec")
    print(f"🧵 Processes used: {max_workers}")
    print(f"💾 Features DB saved: {features_db_path}")
    print(f"📈 Stats saved: {stats_path}")
    
    return features_db


if __name__ == "__main__":
    dataset_path = "dataset"
    output_path = "features"
    
    print("🌸 CBIR Feature Extraction with Multiprocessing")
    print("=" * 60)
    
    features_db = extract_features_from_dataset(dataset_path, output_path, max_workers=4)
    print("\n🎉 Feature extraction completed!")
