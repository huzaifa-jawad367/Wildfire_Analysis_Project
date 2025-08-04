import os
import argparse
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_image_pair(img_filename, img_path, mask_path):
    try:
        # Get image dimensions
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        # Get mask dimensions
        mask = Image.open(mask_path)
        mask_w, mask_h = mask.size
        
        # If dimensions don't match, resize mask to match image
        if img_w != mask_w or img_h != mask_h:
            resized_mask = mask.resize((img_w, img_h), Image.NEAREST)
            resized_mask.save(mask_path)
            return f"Fixed {img_filename}: from {mask_w}x{mask_h} to {img_w}x{img_h}"
        return None
    except Exception as e:
        return f"Error processing {img_filename}: {str(e)}"

def fix_mask_sizes(image_dir, mask_dir):
    """
    Resizes mask images to match their corresponding image dimensions.
    
    Args:
        image_dir: Directory containing the original images
        mask_dir: Directory containing the mask images
    """
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images")
    
    # Create a lookup of mask files
    mask_files = os.listdir(mask_dir)
    
    tasks = []
    
    # Create tasks
    for img_filename in image_files:
        # Find corresponding mask
        mask_name = img_filename.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
        
        if mask_name in mask_files:
            img_path = os.path.join(image_dir, img_filename)
            mask_path = os.path.join(mask_dir, mask_name)
            tasks.append((img_filename, img_path, mask_path))
    
    # Process in parallel
    fixed_count = 0
    errors = []
    
    print(f"Processing {len(tasks)} image-mask pairs...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(lambda x: process_image_pair(*x), tasks),
            total=len(tasks),
            desc="Fixing masks"
        ))
    
    # Count results
    for result in results:
        if result:
            if result.startswith("Error"):
                errors.append(result)
            else:
                fixed_count += 1
    
    print(f"Fixed {fixed_count} masks")
    if errors:
        print(f"Encountered {len(errors)} errors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix mask sizes to match their images")
    parser.add_argument("--image_dir", type=str, default="datasets/iris/images", 
                        help="Directory containing the original images")
    parser.add_argument("--mask_dir", type=str, default="datasets/iris/masks",
                        help="Directory containing the mask images")
    
    args = parser.parse_args()
    fix_mask_sizes(args.image_dir, args.mask_dir) 