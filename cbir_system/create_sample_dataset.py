"""
Create synthetic sample flower images for CBIR system
"""
import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

def create_sample_images():
    """Create synthetic flower-like images with different colors and patterns"""
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Create images with different colors and patterns
    image_configs = [
        # Red flowers
        {"name": "red_flower_1", "primary_color": (220, 50, 50), "secondary_color": (255, 100, 100)},
        {"name": "red_flower_2", "primary_color": (200, 30, 30), "secondary_color": (255, 80, 80)},
        {"name": "red_flower_3", "primary_color": (180, 40, 40), "secondary_color": (255, 120, 120)},
        
        # Blue flowers  
        {"name": "blue_flower_1", "primary_color": (50, 50, 220), "secondary_color": (100, 100, 255)},
        {"name": "blue_flower_2", "primary_color": (30, 30, 200), "secondary_color": (80, 80, 255)},
        {"name": "blue_flower_3", "primary_color": (40, 40, 180), "secondary_color": (120, 120, 255)},
        
        # Yellow flowers
        {"name": "yellow_flower_1", "primary_color": (220, 220, 50), "secondary_color": (255, 255, 100)},
        {"name": "yellow_flower_2", "primary_color": (200, 200, 30), "secondary_color": (255, 255, 80)},
        {"name": "yellow_flower_3", "primary_color": (180, 180, 40), "secondary_color": (255, 255, 120)},
        
        # Purple flowers
        {"name": "purple_flower_1", "primary_color": (150, 50, 150), "secondary_color": (200, 100, 200)},
        {"name": "purple_flower_2", "primary_color": (130, 30, 130), "secondary_color": (180, 80, 180)},
        {"name": "purple_flower_3", "primary_color": (120, 40, 120), "secondary_color": (160, 120, 160)},
        
        # Orange flowers
        {"name": "orange_flower_1", "primary_color": (255, 120, 30), "secondary_color": (255, 160, 80)},
        {"name": "orange_flower_2", "primary_color": (230, 100, 20), "secondary_color": (255, 140, 60)},
        
        # Pink flowers
        {"name": "pink_flower_1", "primary_color": (255, 120, 180), "secondary_color": (255, 160, 200)},
        {"name": "pink_flower_2", "primary_color": (230, 100, 160), "secondary_color": (255, 140, 190)},
    ]
    
    def create_flower_image(config):
        # Create a 256x256 image
        size = (256, 256)
        image = Image.new('RGB', size, (240, 240, 240))  # Light gray background
        draw = ImageDraw.Draw(image)
        
        # Draw flower center
        center = (128, 128)
        # Main flower circle
        draw.ellipse([80, 80, 176, 176], fill=config["primary_color"])
        
        # Add petals (smaller circles around the main one)
        petal_positions = [
            (100, 60), (156, 60),   # top petals
            (180, 100), (180, 156), # right petals  
            (156, 180), (100, 180), # bottom petals
            (76, 156), (76, 100)    # left petals
        ]
        
        for pos in petal_positions:
            draw.ellipse([pos[0]-15, pos[1]-15, pos[0]+15, pos[1]+15], 
                        fill=config["secondary_color"])
        
        # Draw center
        draw.ellipse([118, 118, 138, 138], fill=(255, 255, 100))  # Yellow center
        
        # Add some noise/texture
        for _ in range(100):
            x = np.random.randint(0, 256)
            y = np.random.randint(0, 256)
            color = tuple(np.random.randint(0, 50, 3))
            draw.point((x, y), fill=color)
        
        return image
    
    # Generate all images
    for config in image_configs:
        image = create_flower_image(config)
        filepath = dataset_dir / f"{config['name']}.jpg"
        image.save(filepath, "JPEG", quality=90)
        print(f"Created {filepath}")
    
    print(f"\nDataset created in {dataset_dir.absolute()}")
    print(f"Total images: {len(list(dataset_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    create_sample_images()