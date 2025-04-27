import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.plot import reshape_as_image
import cv2
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from pathlib import Path

def setup_logging():
    """Set up basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

def calculate_indices(image):
    """
    Calculate spectral indices from Sentinel-2 bands
    Sentinel-2 bands typically come in this order:
    - Band 2: Blue
    - Band 3: Green
    - Band 4: Red
    - Band 8: NIR
    - Band 11: SWIR1
    - Band 12: SWIR2
    
    Args:
        image: Numpy array with Sentinel-2 bands
        
    Returns:
        Dictionary of calculated indices
    """
    # Extract bands (assuming common Sentinel-2 band order)
    # Adjust these indices based on your specific band order
    blue = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    red = image[:, :, 2].astype(float)
    nir = image[:, :, 3].astype(float) if image.shape[2] > 3 else None
    swir1 = image[:, :, 4].astype(float) if image.shape[2] > 4 else None
    swir2 = image[:, :, 5].astype(float) if image.shape[2] > 5 else None
    
    indices = {}
    
    # Avoid division by zero
    epsilon = 1e-8
    
    # NDVI (Normalized Difference Vegetation Index) - for vegetation detection
    if nir is not None:
        indices['ndvi'] = np.where(
            (nir + red) > epsilon,
            (nir - red) / (nir + red + epsilon),
            0
        )
    else:
        # Pseudo-NDVI using green instead of NIR
        indices['ndvi'] = np.where(
            (green + red) > epsilon,
            (green - red) / (green + red + epsilon),
            0
        )
    
    # NDWI (Normalized Difference Water Index) - for water detection
    if nir is not None:
        indices['ndwi'] = np.where(
            (nir + green) > epsilon,
            (green - nir) / (green + nir + epsilon),
            0
        )
    else:
        # Pseudo-NDWI using blue and green
        indices['ndwi'] = np.where(
            (blue + green) > epsilon,
            (blue - green) / (blue + green + epsilon),
            0
        )
    
    # NBR (Normalized Burn Ratio) - for burn scar detection
    if nir is not None and swir2 is not None:
        indices['nbr'] = np.where(
            (nir + swir2) > epsilon,
            (nir - swir2) / (nir + swir2 + epsilon),
            0
        )
    
    # NDBI (Normalized Difference Built-up Index) - for urban/built-up areas
    if swir1 is not None and nir is not None:
        indices['ndbi'] = np.where(
            (swir1 + nir) > epsilon,
            (swir1 - nir) / (swir1 + nir + epsilon),
            0
        )
    
    # BSI (Bare Soil Index) - for bare soil/terrain
    if red is not None and green is not None and blue is not None and nir is not None:
        indices['bsi'] = np.where(
            ((red + green) + (nir + red)) > epsilon,
            ((red + green) - (blue + nir)) / ((red + green) + (blue + nir) + epsilon),
            0
        )
    
    # Urban Area Index - specialized for detecting urban areas in RGB images
    # Higher values for bright urban surfaces
    if nir is None:
        indices['urban'] = np.where(
            (red > 0.5) & (green > 0.5) & (blue > 0.5) & 
            (np.abs(red - green) < 0.1) & (np.abs(red - blue) < 0.1),
            1.0,
            0.0
        )
    
    return indices

def create_mask_from_sentinel(image_path, output_path, img_size=512):
    """
    Create a 5-class mask from a Sentinel-2 image
    Classes:
    0: Background
    1: Water
    2: Terrain (bare soil, rocks, urban)
    3: Vegetation (grassland, crops, low vegetation)
    4: Forest (dense tree cover)
    
    Args:
        image_path: Path to the Sentinel-2 image file
        output_path: Path to save the output mask
        img_size: Size to resize the image and mask
    """
    try:
        # Check if image_path is a directory
        if os.path.isdir(image_path):
            # Process all images in directory
            process_sentinel_directory(image_path, os.path.dirname(output_path), img_size)
            return None
            
        # Open the image file
        if image_path.endswith(('.tif', '.TIF')):
            # For GeoTIFF files using rasterio
            with rasterio.open(image_path) as src:
                img = src.read()
                img = reshape_as_image(img)  # Convert to (height, width, bands)
        else:
            # For regular image files using PIL
            img = np.array(Image.open(image_path))
        
        # Calculate indices
        indices = calculate_indices(img)
        
        # Create mask initialized with background class (0)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # Water (Class 1)
        if 'ndwi' in indices:
            water_mask = indices['ndwi'] > 0.2  # Adjust threshold as needed
            mask[water_mask] = 1
        
        # Terrain (Class 2) - bare soil, urban areas
        if 'bsi' in indices:
            terrain_mask = indices['bsi'] > -0.1  # Reduced threshold to catch more urban areas
            # Don't overwrite water
            terrain_mask = terrain_mask & (mask == 0)
            mask[terrain_mask] = 2
        elif 'ndbi' in indices:
            # Use NDBI if BSI is not available
            terrain_mask = indices['ndbi'] > -0.1  # Reduced threshold
            terrain_mask = terrain_mask & (mask == 0)
            mask[terrain_mask] = 2
        elif 'urban' in indices:
            # For RGB images without NIR, use urban index
            terrain_mask = indices['urban'] > 0.5
            terrain_mask = terrain_mask & (mask == 0)
            mask[terrain_mask] = 2
            
            # For RGB urban images, also mark very bright areas as terrain/urban
            if img.shape[2] == 3:
                # Calculate brightness
                brightness = np.mean(img, axis=2)
                bright_mask = brightness > 0.7 * np.max(brightness)
                bright_mask = bright_mask & (mask == 0)
                mask[bright_mask] = 2
        
        # Vegetation (Class 3) - low/sparse vegetation
        veg_mask = (indices['ndvi'] > 0.2) & (indices['ndvi'] <= 0.5)
        veg_mask = veg_mask & (mask == 0)  # Don't overwrite previous classes
        mask[veg_mask] = 3
        
        # Forest (Class 4) - dense vegetation/forest
        forest_mask = indices['ndvi'] > 0.5  # Higher NDVI for forest
        forest_mask = forest_mask & (mask == 0)  # Don't overwrite previous classes
        mask[forest_mask] = 4
        
        # For RGB-only images, detect green areas more aggressively for vegetation/forest
        if img.shape[2] == 3:
            normalized_img = img.astype(float)
            if normalized_img.max() > 1.0:
                normalized_img = normalized_img / 255.0
                
            # Green dominance for vegetation
            r, g, b = normalized_img[:,:,0], normalized_img[:,:,1], normalized_img[:,:,2]
            green_dominance = (g > r*1.1) & (g > b*1.1)
            light_green = green_dominance & (g > 0.3) & (g <= 0.5) & (mask == 0)
            dark_green = green_dominance & (g > 0.1) & (g <= 0.3) & (mask == 0)
            
            mask[light_green] = 3  # Vegetation
            mask[dark_green] = 4   # Forest
        
        # Resize mask to target size
        mask_resized = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save mask as PNG
        Image.fromarray(mask_resized).save(output_path)
        
        return mask_resized
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

def process_sentinel_directory(source_dir, masks_dir, img_size=512):
    """
    Process all Sentinel-2 images in a directory and create masks
    
    Args:
        source_dir: Directory containing Sentinel-2 images
        masks_dir: Directory to save the masks
        img_size: Size to resize the images and masks
    """
    setup_logging()
    
    # Ensure directories exist
    source_dir = Path(source_dir)
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['.tif', '.TIF', '.jp2', '.jpg', '.png']:
        image_files.extend(list(source_dir.glob(f"*{ext}")))
    
    if not image_files:
        logging.error(f"No images found in {source_dir}")
        return
    
    logging.info(f"Found {len(image_files)} images in {source_dir}")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Creating masks"):
        # Create output mask path
        mask_path = masks_dir / f"{img_path.stem}_mask.png"
        
        # Process image and create mask
        create_mask_from_sentinel(img_path, mask_path, img_size)
    
    logging.info(f"Processed {len(image_files)} images. Masks saved to {masks_dir}")

def visualize_sample(image_path, mask_path, output_path):
    """
    Create a visualization of an image and its mask side by side
    
    Args:
        image_path: Path to the image
        mask_path: Path to the mask
        output_path: Path to save the visualization
    """
    # Read the image and mask
    if image_path.endswith(('.tif', '.TIF')):
        with rasterio.open(image_path) as src:
            img = src.read()
            img = reshape_as_image(img)
            
            # If more than 3 bands, use only RGB bands (2,3,4 for Sentinel-2)
            if img.shape[2] > 3:
                # Get RGB equivalent bands (adjust indices based on your band order)
                img = img[:, :, 0:3]
            
            # Normalize for visualization
            for i in range(min(3, img.shape[2])):
                p2 = np.percentile(img[:, :, i], 2)
                p98 = np.percentile(img[:, :, i], 98)
                if p98 > p2:
                    img[:, :, i] = np.clip((img[:, :, i] - p2) / (p98 - p2), 0, 1)
            
            # Convert to 8-bit
            img = (img * 255).astype(np.uint8)
    else:
        img = np.array(Image.open(image_path))
    
    # Read mask
    mask = np.array(Image.open(mask_path))
    
    # Create a color-coded mask for visualization
    colors = [
        [0, 0, 0],        # Background - Black
        [0, 0, 255],      # Water - Blue
        [139, 69, 19],    # Terrain - Brown
        [0, 255, 0],      # Vegetation - Green
        [0, 100, 0]       # Forest - Dark Green
    ]
    
    mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        mask_colored[mask == class_idx] = color
    
    # Create the visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Sentinel-2 Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_colored)
    plt.title('Generated Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # Create a semi-transparent overlay
    overlay = img.copy()
    for class_idx, color in enumerate(colors):
        if class_idx > 0:  # Skip background
            class_mask = mask == class_idx
            overlay[class_mask] = (
                0.7 * img[class_mask] + 
                0.3 * np.array(color, dtype=np.uint8)
            )
    
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    # Add legend
    class_names = ['Background', 'Water', 'Terrain', 'Vegetation', 'Forest']
    patches = [plt.Rectangle((0, 0), 1, 1, fc=np.array(color)/255) for color in colors]
    plt.legend(patches, class_names, loc='lower right', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create masks from Sentinel-2 images")
    parser.add_argument("source_dir", help="Directory containing Sentinel-2 images or path to a single image")
    parser.add_argument("--masks_dir", default="datasets/sentinel2/masks", 
                      help="Directory to save masks (default: datasets/sentinel2/masks)")
    parser.add_argument("--img_size", type=int, default=512, 
                      help="Size to resize images and masks (default: 512)")
    parser.add_argument("--visualize", action="store_true", 
                      help="Create visualizations of the masks")
    parser.add_argument("--viz_output", default="datasets/sentinel2/visualizations", 
                      help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Check if source_dir is a single file or directory
    if os.path.isfile(args.source_dir):
        # Process single file
        file_name = os.path.basename(args.source_dir)
        output_path = os.path.join(args.masks_dir, os.path.splitext(file_name)[0] + "_mask.png")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process image and create mask
        mask = create_mask_from_sentinel(args.source_dir, output_path, args.img_size)
        
        # Create visualization if requested
        if args.visualize and mask is not None:
            os.makedirs(args.viz_output, exist_ok=True)
            viz_path = os.path.join(args.viz_output, os.path.splitext(file_name)[0] + "_viz.png")
            visualize_sample(args.source_dir, output_path, viz_path)
            
        logging.info(f"Processed single image. Mask saved to {output_path}")
    else:
        # Process directory
        process_sentinel_directory(args.source_dir, args.masks_dir, args.img_size)
    
        # Create visualizations if requested
        if args.visualize:
            viz_dir = Path(args.viz_output)
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            source_dir = Path(args.source_dir)
            masks_dir = Path(args.masks_dir)
            
            # Find all mask files
            mask_files = list(masks_dir.glob("*_mask.png"))
            
            logging.info(f"Creating visualizations for {len(mask_files)} images")
            
            for mask_path in tqdm(mask_files, desc="Creating visualizations"):
                # Get corresponding image file
                img_stem = mask_path.stem.replace("_mask", "")
                
                # Try to find matching image with different extensions
                img_path = None
                for ext in ['.tif', '.TIF', '.jp2', '.jpg', '.png']:
                    candidate = source_dir / f"{img_stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if img_path:
                    viz_path = viz_dir / f"{img_stem}_viz.png"
                    visualize_sample(img_path, mask_path, viz_path)
            
            logging.info(f"Visualizations saved to {viz_dir}") 