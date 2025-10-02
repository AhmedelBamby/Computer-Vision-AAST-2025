"""
Download and setup Kaggle PyTorch Challenge Flower Dataset with multithreading
"""
import os
import zipfile
import shutil
import requests
from pathlib import Path
import concurrent.futures
from threading import Lock
import time
import json

def download_kaggle_dataset():
    """Download the Kaggle flower dataset"""
    print("📥 Downloading Kaggle PyTorch Challenge Flower Dataset...")
    
    # Create downloads directory
    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(exist_ok=True)
    
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/nunenuh/pytorch-challange-flower-dataset"
    zip_path = downloads_dir / "pytorch-challange-flower-dataset.zip"
    
    try:
        # Download with progress
        print(f"Downloading from: {dataset_url}")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
        
        print(f"\n✅ Downloaded: {zip_path}")
        return zip_path
        
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return None

def extract_dataset(zip_path, target_dir):
    """Extract the dataset to target directory"""
    print(f"\n📂 Extracting dataset to {target_dir}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        print("✅ Dataset extracted successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to extract: {e}")
        return False

def organize_flower_dataset(extract_dir, dataset_dir):
    """Organize the flower dataset structure"""
    print(f"\n🗂️  Organizing dataset structure...")
    
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for the extracted folder
    extract_path = Path(extract_dir)
    
    # Find the actual dataset folder (it might be nested)
    dataset_folders = []
    for root, dirs, files in os.walk(extract_path):
        for dir_name in dirs:
            if any(flower_type in dir_name.lower() for flower_type in ['flower', 'train', 'test', 'valid']):
                dataset_folders.append(Path(root) / dir_name)
    
    if not dataset_folders:
        # Look for image files directly
        for root, dirs, files in os.walk(extract_path):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                dataset_folders.append(Path(root))
                break
    
    if not dataset_folders:
        print("❌ No dataset folder found in extracted files")
        return False
    
    source_dir = dataset_folders[0]
    print(f"📁 Found dataset in: {source_dir}")
    
    # Copy organized structure
    try:
        if source_dir != dataset_dir:
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            shutil.copytree(source_dir, dataset_dir)
        
        print(f"✅ Dataset organized in: {dataset_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to organize dataset: {e}")
        return False

def count_images_parallel(dataset_dir, max_workers=8):
    """Count images in dataset using parallel processing"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        return {}
    
    # Get all subdirectories (categories)
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    def count_category_images(category_path):
        """Count images in a single category"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        count = 0
        images = []
        
        try:
            for file_path in category_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    count += 1
                    images.append(file_path)
            
            return category_path.name, count, images
        except Exception as e:
            print(f"❌ Error counting images in {category_path}: {e}")
            return category_path.name, 0, []
    
    # Count images in parallel
    category_stats = {}
    total_images = 0
    
    print(f"\n📊 Counting images using {max_workers} threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_category = {executor.submit(count_category_images, cat): cat for cat in categories}
        
        for future in concurrent.futures.as_completed(future_to_category):
            category_path = future_to_category[future]
            try:
                category_name, count, images = future.result()
                category_stats[category_name] = {
                    'count': count,
                    'images': [str(img.relative_to(dataset_path)) for img in images]
                }
                total_images += count
                print(f"✅ {category_name}: {count} images")
                
            except Exception as e:
                print(f"❌ Error processing {category_path}: {e}")
    
    print(f"\n🎉 Total images found: {total_images}")
    return category_stats, total_images

def create_sample_dataset():
    """Create a sample dataset if download fails"""
    print("🎨 Creating sample flower dataset...")
    
    from PIL import Image, ImageDraw
    import random
    import numpy as np
    
    dataset_dir = Path("dataset")
    
    # Flower categories inspired by common flower datasets
    categories = {
        'daisy': {'colors': [(255, 255, 100), (255, 255, 150), (240, 240, 80)]},
        'dandelion': {'colors': [(255, 200, 50), (255, 220, 80), (240, 180, 30)]},
        'rose': {'colors': [(255, 100, 100), (255, 150, 150), (220, 80, 80)]},
        'sunflower': {'colors': [(255, 200, 50), (255, 180, 30), (240, 160, 20)]},
        'tulip': {'colors': [(255, 100, 150), (200, 80, 255), (150, 100, 200)]}
    }
    
    images_per_category = 10
    
    for category, config in categories.items():
        category_dir = dataset_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(images_per_category):
            # Create flower image
            size = (224, 224)  # Standard size
            image = Image.new('RGB', size, (220, 255, 220))  # Light green background
            draw = ImageDraw.Draw(image)
            
            # Draw flower
            center = (112, 112)
            petal_color = random.choice(config['colors'])
            
            # Draw petals
            for j in range(8):  # 8 petals
                angle = j * 45
                angle_rad = np.radians(angle)
                petal_x = center[0] + int(40 * np.cos(angle_rad))
                petal_y = center[1] + int(40 * np.sin(angle_rad))
                
                # Draw petal
                petal_coords = [
                    petal_x - 15, petal_y - 15,
                    petal_x + 15, petal_y + 15
                ]
                draw.ellipse(petal_coords, fill=petal_color)
            
            # Draw center
            center_coords = [center[0] - 8, center[1] - 8, center[0] + 8, center[1] + 8]
            draw.ellipse(center_coords, fill=(255, 255, 100))
            
            # Add noise
            for _ in range(50):
                x = random.randint(0, 223)
                y = random.randint(0, 223)
                noise_color = tuple(random.randint(0, 20) for _ in range(3))
                draw.point((x, y), fill=noise_color)
            
            # Save image
            filename = f"{category}_{i+1:03d}.jpg"
            filepath = category_dir / filename
            image.save(filepath, "JPEG", quality=85)
        
        print(f"✅ Created {images_per_category} images for {category}")
    
    return dataset_dir

def setup_kaggle_flower_dataset():
    """Main function to setup the Kaggle flower dataset"""
    print("🌸 Kaggle PyTorch Challenge Flower Dataset Setup")
    print("=" * 60)
    
    # Try to download the Kaggle dataset
    zip_path = download_kaggle_dataset()
    
    if zip_path and zip_path.exists():
        # Extract dataset
        extract_dir = Path("temp_extract")
        if extract_dataset(zip_path, extract_dir):
            # Organize dataset
            if organize_flower_dataset(extract_dir, "dataset"):
                # Clean up
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                
                # Count images
                stats, total = count_images_parallel("dataset", max_workers=8)
                
                # Save dataset info
                dataset_info = {
                    'source': 'Kaggle PyTorch Challenge Flower Dataset',
                    'total_images': total,
                    'categories': stats,
                    'download_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                with open('dataset/dataset_info.json', 'w') as f:
                    json.dump(dataset_info, f, indent=2)
                
                print(f"\n🎉 Kaggle dataset setup complete!")
                print(f"📁 Dataset location: dataset/")
                print(f"📊 Total images: {total}")
                
                return True
    
    # Fallback to sample dataset
    print("\n⚠️  Kaggle download failed, creating sample dataset...")
    sample_dir = create_sample_dataset()
    
    # Count sample images
    stats, total = count_images_parallel(sample_dir, max_workers=8)
    
    # Save dataset info
    dataset_info = {
        'source': 'Generated Sample Flower Dataset',
        'total_images': total,
        'categories': stats,
        'download_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('dataset/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n🎉 Sample dataset created!")
    print(f"📁 Dataset location: dataset/")
    print(f"📊 Total images: {total}")
    
    return True

if __name__ == "__main__":
    setup_kaggle_flower_dataset()