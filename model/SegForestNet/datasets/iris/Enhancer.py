import os
import re
import ee
import glob
import time
import datetime
import requests
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

# Initialize Earth Engine
try:
    ee.Initialize(project="fiery-cistern-454011-e3")
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    error_msg = (
        f"Error initializing Earth Engine: {str(e)}\n"
        "Please make sure you have:\n"
        "1. Installed the earthengine-api package\n"
        "2. Authenticated with Earth Engine using 'earthengine authenticate'\n"
        "3. Have proper credentials and permissions for the project"
    )
    print(error_msg)
    import sys
    sys.exit(1)

def extract_info_from_filename(filename):
    """Extract satellite type, latitude, longitude, and date from existing filenames."""
    # For files with our standard naming convention (from the Sentinel_2EXT.py script)
    if filename.startswith(('S2_', 'L8_')):
        # Pattern: S2_lat45.7462_lon6.0538_2024-11-27_veg34.2.png
        # Modified regex to handle possible trailing period in vegetation score and possible hyphen in veg score
        pattern = r'([A-Z0-9]+)_lat([-\d\.]+)_lon([-\d\.]+)_(\d{4}-\d{2}-\d{2})_veg(-?[\d\.]+)\.?'
        match = re.match(pattern, filename)
        
        if match:
            satellite = match.group(1)  # 'S2' or 'L8'
            latitude = float(match.group(2))
            longitude = float(match.group(3))
            date = match.group(4)
            # Remove any trailing periods from the vegetation score
            veg_score_str = match.group(5).rstrip('.')
            veg_score = float(veg_score_str)
            
            return {
                "satellite": satellite,
                "latitude": latitude,
                "longitude": longitude,
                "date": date,
                "veg_score": veg_score
            }
    
    # For other types of files that might exist with coordinates/dates hidden elsewhere
    # Extract coordinates and date from files like: 20240224_152114_87_24fc_3B_AnalyticMS_SR_PlanetScope_Calamar_Guaviare
    # Will set default coordinates for Calamar, Guaviare, Colombia since it's mentioned in filename
    if "Calamar_Guaviare" in filename:
        date_match = re.search(r'(\d{8})_', filename)
        date_str = date_match.group(1) if date_match else "20240224"
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # Default coordinates for Calamar, Guaviare in Colombia
        return {
            "satellite": "S2",  # Default to Sentinel-2
            "latitude": 2.0912,  # Approximate coordinates for Calamar, Guaviare
            "longitude": -72.6539,
            "date": formatted_date,
            "veg_score": 50.0  # Default vegetation score
        }
    
    # If we can't extract information, return None
    return None

def get_best_satellite_image(latitude, longitude, date_str, buffer_km=5, days_range=15, max_cloud_percent=20):
    """
    Fetch the best satellite image (with lowest cloud coverage) from Google Earth Engine.
    """
    print(f"Fetching best satellite image for coordinates ({latitude:.4f}, {longitude:.4f}) around {date_str}...")

    # Create a point geometry
    point = ee.Geometry.Point([longitude, latitude])

    # Create a buffer around the point
    region = point.buffer(buffer_km * 1000)

    # Parse the date
    target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    # Convert target date to milliseconds since epoch for Earth Engine
    target_timestamp = int(target_date.timestamp() * 1000)

    # Define the date range
    start_date = target_date - timedelta(days=days_range)
    end_date = target_date + timedelta(days=days_range)

    # Format dates for Earth Engine
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Get Sentinel-2 collection
    collection = (ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start_date_str, end_date_str)
        .filterBounds(region))

    # Add Landsat 8 collection
    landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(start_date_str, end_date_str)
        .filterBounds(region))

    def add_sentinel_cloud_score(img):
        # Calculate cloud score for the region of interest using Sentinel-2's cloud probability band
        cloud_score = img.select(['MSK_CLDPRB']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,  # Scale in meters
            maxPixels=1e9
        ).get('MSK_CLDPRB')
        
        # Calculate NDVI to measure vegetation greenness
        ndvi = img.normalizedDifference(['B8', 'B4'])
        mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,
            maxPixels=1e9
        ).get('nd')
        
        # Calculate vegetation score (higher is better)
        # Convert to a number between 0 and 100
        veg_score = ee.Number(mean_ndvi).multiply(100)
        
        # Add date difference from target date
        date_diff = ee.Number(img.get('system:time_start')).subtract(target_timestamp).abs()
        
        # Convert cloud score and vegetation score to ensure proper weighting
        cloud_score = ee.Number(cloud_score)
        veg_score = ee.Number(veg_score)
        
        # Lower score is better, so we create a combined score
        # that prioritizes low cloud cover and high vegetation, but also considers date proximity
        # Note that veg_score is subtracted because higher vegetation is preferred
        combined_score = cloud_score.multiply(0.5) \
            .subtract(veg_score.multiply(0.3)) \
            .add(ee.Number(date_diff).divide(1000 * 60 * 60 * 24).multiply(0.2))
        
        return img.set({
            'cloud_score': cloud_score,
            'veg_score': veg_score,
            'date_diff': date_diff,
            'combined_score': combined_score,
            'satellite': 'S2'
        })

    def add_landsat_cloud_score(img):
        # For Landsat 8, use the QA_PIXEL band (bit 3 is cloud)
        qa_band = img.select('QA_PIXEL')
        cloud_bit = 1 << 3
        cloud_mask = qa_band.bitwiseAnd(cloud_bit).eq(0)
        
        # Calculate percentage of clear pixels
        clear_pct = cloud_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).get('QA_PIXEL')
        
        # Convert to cloud percentage (100 - clear_pct)
        # Use ee.Number for calculation to keep everything server-side
        cloud_score = ee.Number(100).subtract(ee.Number(clear_pct).multiply(100))
        
        # Calculate NDVI to measure vegetation greenness (Landsat 8 bands)
        ndvi = img.normalizedDifference(['SR_B5', 'SR_B4'])
        mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).get('nd')
        
        # Calculate vegetation score (higher is better)
        # Convert to a number between 0 and 100
        veg_score = ee.Number(mean_ndvi).multiply(100)
        
        # Add date difference from target date
        date_diff = ee.Number(img.get('system:time_start')).subtract(target_timestamp).abs()
        
        # Lower score is better, so we create a combined score
        # that prioritizes low cloud cover and high vegetation, but also considers date proximity
        combined_score = ee.Number(cloud_score).multiply(0.5) \
            .subtract(veg_score.multiply(0.3)) \
            .add(ee.Number(date_diff).divide(1000 * 60 * 60 * 24).multiply(0.2))
        
        return img.set({
            'cloud_score': cloud_score,
            'veg_score': veg_score,
            'date_diff': date_diff,
            'combined_score': combined_score,
            'satellite': 'L8'
        })

    # Add cloud scores to all images
    s2_with_scores = collection.map(add_sentinel_cloud_score)
    l8_with_scores = landsat.map(add_landsat_cloud_score)
    
    # Merge collections
    all_images = s2_with_scores.merge(l8_with_scores)
    
    # Filter by cloud cover
    filtered_images = all_images.filter(ee.Filter.lte('cloud_score', max_cloud_percent))
    
    # Sort by combined score (ascending) to get the best image first
    sorted_images = filtered_images.sort('combined_score')
    
    # Get first image (with best combined score)
    try:
        best_image = ee.Image(sorted_images.first())
        
        # Check if we actually got an image
        image_id = best_image.get('system:id').getInfo()
        
        if not image_id:
            print("  No suitable images found for this location.")
            return None, None
            
        cloud_score = best_image.get('cloud_score').getInfo()
        veg_score = best_image.get('veg_score').getInfo()
        date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        satellite = best_image.get('satellite').getInfo()
        date_diff = best_image.get('date_diff').getInfo()
        date_diff_days = date_diff / (1000 * 60 * 60 * 24)
        
        print(f"  Best image found: Date={date}, Satellite={satellite}, Cloud Score={cloud_score:.1f}%")
        print(f"  Vegetation Score: {veg_score:.1f}/100 (higher is better)")
        print(f"  Time difference from target: {date_diff_days:.1f} days")
        
        return {
            'image': best_image,
            'cloud_score': cloud_score,
            'veg_score': veg_score,
            'date': date,
            'satellite': satellite
        }, region
        
    except Exception as e:
        print(f"  Error finding best image: {e}")
        return None, None

def download_image(image_info, region, latitude, longitude, original_filename, output_dir="datasets/iris/images"):
    """Download an image with natural color visualization and generate masks"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        masks_dir = os.path.join(os.path.dirname(output_dir), "masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        image = image_info['image']
        satellite = image_info['satellite']
        
        # Use the original filename as base, or create a new one if needed
        if original_filename.endswith('.png') or original_filename.endswith('.jpg'):
            # Keep the original base name
            base_filename = os.path.splitext(original_filename)[0]
        else:
            # Generate a filename with satellite platform info
            base_filename = f"{satellite}_lat{latitude:.4f}_lon{longitude:.4f}_{image_info['date']}_veg{image_info['veg_score']:.1f}"
        
        image_filename = f"{output_dir}/{base_filename}.png"
        mask_filename = f"{masks_dir}/{base_filename}_mask.png"
        
        # Define visualization parameters for natural colors
        if satellite == "S2":  # Sentinel-2
            # Apply a conservative cloud mask to improve image quality
            cloud_prob_threshold = 60  # Higher threshold for less aggressive masking
            cloud_mask = image.select('MSK_CLDPRB').lt(cloud_prob_threshold)
            image = image.updateMask(cloud_mask)
            
            # Natural true color visualization for Sentinel-2
            vizParams = {
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 2000,
                'gamma': 1.1  # Slightly enhanced gamma for better vegetation display
            }
            
            # Calculate NDVI for mask generation
            ndvi = image.normalizedDifference(['B8', 'B4'])
            
        else:  # Landsat 8
            # Apply a conservative cloud mask for Landsat
            cloud_bit = 1 << 3
            cloud_mask = image.select('QA_PIXEL').bitwiseAnd(cloud_bit).eq(0)
            image = image.updateMask(cloud_mask)
            
            # Natural true color visualization for Landsat 8
            vizParams = {
                'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
                'min': 5000,
                'max': 15000,
                'gamma': 1.1  # Slightly enhanced gamma for better vegetation display
            }
            
            # Calculate NDVI for mask generation
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4'])
        
        # Create a 5-class land cover mask using spectral indices
        # 0: Background
        # 1: Water
        # 2: Terrain
        # 3: Vegetation
        # 4: Forest
        
        # Use spectral indices to classify the image
        if satellite == "S2":
            # Water - NDWI (Normalized Difference Water Index)
            ndwi = image.normalizedDifference(['B3', 'B8'])
            water = ndwi.gt(0.2)
            
            # Vegetation thresholds (NDVI-based)
            veg_low = ndvi.gt(0.3)
            forest = ndvi.gt(0.6)  # Dense vegetation / forest
            vegetation = veg_low.subtract(forest)  # Intermediate vegetation
            
            # Non-vegetated terrain
            terrain = ndvi.lt(0.3).And(ndwi.lt(0.2))
            
        else:  # Landsat 8
            # Water - NDWI (modified for Landsat)
            ndwi = image.normalizedDifference(['SR_B3', 'SR_B5'])
            water = ndwi.gt(0.2)
            
            # Vegetation thresholds (NDVI-based)
            veg_low = ndvi.gt(0.3)
            forest = ndvi.gt(0.6)  # Dense vegetation / forest
            vegetation = veg_low.subtract(forest)  # Intermediate vegetation
            
            # Non-vegetated terrain
            terrain = ndvi.lt(0.3).And(ndwi.lt(0.2))
        
        # Create the 5-class mask
        # Start with zeros
        mask = ee.Image(0)
        # Add each class with its value
        mask = mask.where(water, 1)
        mask = mask.where(terrain, 2)
        mask = mask.where(vegetation, 3)
        mask = mask.where(forest, 4)
        
        # Visualize the mask with colors for each class
        mask_viz = {
            'min': 0,
            'max': 4,
            'palette': ['000000', '0000FF', 'A0522D', '00FF00', '004D00']  # Black, Blue, Brown, Green, Dark Green
        }
        
        try:
            # Download the RGB image at 1024x1024 resolution
            url = image.visualize(**vizParams).getThumbURL({
                'dimensions': 1024,
                'region': region,
                'format': 'png',
            })
            
            # Download the image
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(image_filename, 'wb') as f:
                    f.write(response.content)
                
                print(f"  Downloaded RGB image: {image_filename}")
            else:
                print(f"  Failed to download RGB image. Status code: {response.status_code}")
                return False
                
            # Download the visualization of the mask at 1024x1024
            mask_url = mask.visualize(**mask_viz).getThumbURL({
                'dimensions': 1024,
                'region': region,
                'format': 'png',
            })
            
            # Download the mask visualization
            mask_response = requests.get(mask_url)
            
            if mask_response.status_code == 200:
                with open(mask_filename, 'wb') as f:
                    f.write(mask_response.content)
                
                print(f"  Downloaded mask: {mask_filename}")
                
                # Update train.txt file with the new image filename
                with open("datasets/iris/train.txt", "a") as f:
                    f.write(f"{base_filename}.png\n")
                
                return True
            else:
                print(f"  Failed to download mask. Status code: {mask_response.status_code}")
                return False
                
        except Exception as e:
            print(f"  Error downloading images: {str(e)}")
            return False
            
    except Exception as e:
        print(f"  Error downloading image: {str(e)}")
        return False

def process_image(filename, max_cloud_percent=20):
    """Process a single image from its filename information."""
    image_info = extract_info_from_filename(filename)
    
    if not image_info:
        print(f"Could not extract information from filename: {filename}")
        return False
    
    print(f"\nProcessing {filename}...")
    print(f"Extracted info: {image_info['satellite']} at ({image_info['latitude']:.4f}, {image_info['longitude']:.4f}) from {image_info['date']}")
    
    try:
        # Get the best satellite image with cloud coverage below threshold
        image_data, region = get_best_satellite_image(
            image_info['latitude'], 
            image_info['longitude'], 
            image_info['date'], 
            buffer_km=5,
            days_range=15,
            max_cloud_percent=max_cloud_percent
        )
        
        if image_data:
            # Download the image with natural colors
            success = download_image(image_data, region, image_info['latitude'], image_info['longitude'], filename)
            return success
        else:
            print(f"  Could not find suitable images for {filename}.")
            return False
        
    except Exception as e:
        print(f"  Error processing image {filename}: {str(e)}")
        return False

def process_all_images(max_cloud_percent=20, max_workers=4):
    """Process all images in the images directory."""
    print("Starting image redownload process...")
    
    # Get a list of all image files
    image_dir = "datasets/iris/images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Initialize train.txt - clear it first 
    with open("datasets/iris/train.txt", "w") as f:
        pass
    
    # Initialize val.txt - clear it first
    with open("datasets/iris/val.txt", "w") as f:
        pass
    
    # Get all image files
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif"]:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    # Get just the filenames
    image_filenames = [os.path.basename(f) for f in image_files]
    
    if not image_filenames:
        print("No image files found in the directory.")
        return
    
    print(f"Found {len(image_filenames)} image files to process.")
    
    # Process the images in parallel
    successful_downloads = 0
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, filename, max_cloud_percent) for filename in image_filenames]
            
            processed_files = []
            for i, future in enumerate(futures):
                if future.result():
                    successful_downloads += 1
                    processed_files.append(image_filenames[i])
        
        # Now, let's split the data
        if successful_downloads > 0:
            import random
            random.shuffle(processed_files)
            
            # Split data into train and validation sets
            val_ratio = 0.2  # 20% for validation
            val_count = int(successful_downloads * val_ratio)
            
            if val_count > 0:
                # Read all files from train.txt
                with open("datasets/iris/train.txt", "r") as f:
                    all_files = f.read().splitlines()
                
                if all_files:
                    train_files = []
                    val_files = []
                    
                    # Decide which files go to validation (about 20%)
                    for i, file in enumerate(all_files):
                        if i % 5 == 0:  # Every 5th file
                            val_files.append(file)
                        else:
                            train_files.append(file)
                    
                    # Write train.txt
                    with open("datasets/iris/train.txt", "w") as f:
                        for file in train_files:
                            f.write(f"{file}\n")
                    
                    # Write val.txt
                    with open("datasets/iris/val.txt", "w") as f:
                        for file in val_files:
                            f.write(f"{file}\n")
                    
                    print(f"\nSplit data into {len(train_files)} training and {len(val_files)} validation samples.")
    
    except KeyboardInterrupt:
        print("\nDownload process interrupted by user.")
    
    print(f"\nFinal statistics: {successful_downloads} images downloaded from {len(image_filenames)} attempted locations.")

if __name__ == "__main__":
    # Define max cloud percentage and number of workers
    max_cloud_percent = 20
    max_workers = 4
    
    # Process all images
    process_all_images(max_cloud_percent=max_cloud_percent, max_workers=max_workers) 