"""
Download a sample flower dataset for CBIR system
"""
import os
import requests
from pathlib import Path
import urllib.request

def download_sample_images():
    """Download sample flower images from URLs"""
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Sample flower images (you can replace these with any flower image URLs)
    flower_urls = [
        # Roses
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Rosa_rubiginosa_1.jpg/300px-Rosa_rubiginosa_1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Pink_rose_02.jpg/300px-Pink_rose_02.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Damask_rose.jpg/300px-Damask_rose.jpg",
        
        # Sunflowers
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Sunflower_sky_backdrop.jpg/300px-Sunflower_sky_backdrop.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Helianthus_annuus_20100718_02.jpg/300px-Helianthus_annuus_20100718_02.jpg",
        
        # Tulips
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Red_tulip.jpg/300px-Red_tulip.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Yellow_tulips.jpg/300px-Yellow_tulips.jpg",
        
        # Daisies
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Marguerite_Daisy.jpg/300px-Marguerite_Daisy.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/Bellis_perennis_white_%28aka%29.jpg/300px-Bellis_perennis_white_%28aka%29.jpg",
        
        # Dandelions
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Dandelion.jpg/300px-Dandelion.jpg",
    ]
    
    for i, url in enumerate(flower_urls):
        try:
            filename = f"flower_{i+1:02d}.jpg"
            filepath = dataset_dir / filename
            
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded {filename}")
            
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    
    print(f"\nDataset downloaded to {dataset_dir.absolute()}")
    print(f"Total images: {len(list(dataset_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    download_sample_images()