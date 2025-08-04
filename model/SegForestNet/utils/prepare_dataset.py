import os
import shutil
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

def create_mask_from_image(image_path, num_classes, output_path):
    """
    Helper function to create a sample mask from an image.
    This is just an example - you should modify this based on your segmentation needs.
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a simple mask (this is just an example)
    # You should replace this with your actual mask creation logic
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Example: Simple thresholding-based segmentation
    # Modify this based on your needs
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for i in range(1, num_classes):
        threshold = (255 // num_classes) * i
        mask[gray > threshold] = i
    
    # Save mask
    cv2.imwrite(str(output_path), mask)
    return mask

def prepare_dataset(source_dir, num_classes=6, split_ratio=(0.8, 0.1, 0.1)):
    """
    Prepare dataset for SegForestNet training.
    
    Args:
        source_dir: Directory containing source images
        num_classes: Number of segmentation classes
        split_ratio: (train, val, test) split ratios
    """
    setup_logging()
    
    # Setup paths
    base_dir = Path("datasets/iris")  # Simplified path
    images_dir = base_dir / "images"
    masks_dir = base_dir / "masks"
    samples_dir = base_dir / "samples"
    
    # Ensure directories exist
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images from source directory
    source_dir = Path(source_dir)
    image_files = list(source_dir.glob("*.jpg")) + \
                 list(source_dir.glob("*.jpeg")) + \
                 list(source_dir.glob("*.png"))
    
    if not image_files:
        logging.error(f"No images found in {source_dir}")
        return
    
    logging.info(f"Found {len(image_files)} images")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Read and validate image
            img = Image.open(img_path)
            img = img.convert('RGB')
            
            # Resize to 256x256
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            # Save processed image
            new_img_path = images_dir / img_path.name
            img.save(new_img_path)
            
            # Create and save mask
            mask_path = masks_dir / f"{img_path.stem}_mask.png"
            create_mask_from_image(new_img_path, num_classes, mask_path)
            
        except Exception as e:
            logging.error(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Create train/val/test splits
    all_images = [f.name for f in images_dir.glob("*.jpg")] + \
                [f.name for f in images_dir.glob("*.jpeg")] + \
                [f.name for f in images_dir.glob("*.png")]
    
    np.random.shuffle(all_images)
    
    n_train = int(len(all_images) * split_ratio[0])
    n_val = int(len(all_images) * split_ratio[1])
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    # Save splits
    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images)
    ]:
        with open(base_dir / f"{split_name}.txt", "w") as f:
            f.write("\n".join(split_images))
    
    # Create sample visualizations
    for i, img_name in enumerate(np.random.choice(all_images, 5)):
        img_path = images_dir / img_name
        mask_path = masks_dir / f"{Path(img_name).stem}_mask.png"
        
        # Create side-by-side visualization
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        # Create combined image
        combined = Image.new('RGB', (512, 256))
        combined.paste(img, (0, 0))
        # Convert mask to RGB for visualization
        mask_rgb = Image.fromarray(np.uint8(plt.cm.tab20(np.array(mask)/num_classes)*255))
        combined.paste(mask_rgb, (256, 0))
        
        combined.save(samples_dir / f"sample_{i+1}.png")
    
    logging.info(f"""
Dataset preparation completed:
- Total images: {len(all_images)}
- Training images: {len(train_images)}
- Validation images: {len(val_images)}
- Test images: {len(test_images)}
- Sample visualizations: 5 (saved in samples directory)

Directory structure:
{base_dir}/
├── images/          ({len(all_images)} files)
├── masks/           ({len(all_images)} files)
├── samples/         (5 files)
├── train.txt        ({len(train_images)} files)
├── val.txt          ({len(val_images)} files)
└── test.txt         ({len(test_images)} files)
    """)

def verify_dataset(base_dir):
    """
    Verify the prepared dataset for common issues.
    Returns (bool, str) tuple of (is_valid, message)
    """
    try:
        base_dir = Path(base_dir)
        images_dir = base_dir / "images"
        masks_dir = base_dir / "masks"
        
        # Check directories exist
        if not images_dir.exists() or not masks_dir.exists():
            return False, "Missing images or masks directory"
            
        # Check for image-mask pairs
        image_files = list(images_dir.glob("*.jpg")) + \
                     list(images_dir.glob("*.jpeg")) + \
                     list(images_dir.glob("*.png"))
                     
        if not image_files:
            return False, "No images found"
            
        for img_path in image_files:
            mask_path = masks_dir / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                return False, f"Missing mask for {img_path.name}"
                
            # Verify image can be opened
            try:
                img = Image.open(img_path)
                img.verify()
                if img.size != (256, 256):
                    return False, f"Image {img_path.name} is not 256x256"
            except Exception as e:
                return False, f"Invalid image {img_path.name}: {str(e)}"
                
            # Verify mask
            try:
                mask = Image.open(mask_path)
                mask.verify()
                if mask.size != (256, 256):
                    return False, f"Mask {mask_path.name} is not 256x256"
            except Exception as e:
                return False, f"Invalid mask {mask_path.name}: {str(e)}"
        
        # Check split files exist and are valid
        for split in ['train.txt', 'val.txt', 'test.txt']:
            split_path = base_dir / split
            if not split_path.exists():
                return False, f"Missing {split} file"
            
            with open(split_path) as f:
                files = f.read().splitlines()
                if not files:
                    return False, f"Empty {split} file"
                
                for fname in files:
                    if not (images_dir / fname).exists():
                        return False, f"Image {fname} in {split} doesn't exist"
        
        return True, "Dataset verification passed"
        
    except Exception as e:
        return False, f"Verification failed: {str(e)}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset for SegForestNet")
    parser.add_argument("source_dir", help="Directory containing source images")
    parser.add_argument("--num-classes", type=int, default=6, help="Number of segmentation classes")
    parser.add_argument("--verify", action="store_true", help="Verify existing dataset instead of preparing new one")
    args = parser.parse_args()
    
    if args.verify:
        is_valid, message = verify_dataset("datasets/iris")
        if is_valid:
            print("✓", message)
        else:
            print("✗", message)
            sys.exit(1)
    else:
        prepare_dataset(args.source_dir, args.num_classes) 