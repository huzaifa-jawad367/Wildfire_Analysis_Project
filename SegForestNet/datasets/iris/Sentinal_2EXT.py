import ee
import time
import random
import datetime
import os
import sys
import requests
from datetime import timedelta, datetime
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
    sys.exit(1)

def get_forest_vegetation_regions():
    """Get regions with high concentration of forests and vegetation."""
    # List of coordinates with high vegetation/forest concentration
    forest_regions = [
        # Amazon Rainforest
        {"min_lat": -10, "max_lat": 5, "min_lon": -75, "max_lon": -50},
        # Congo Basin
        {"min_lat": -5, "max_lat": 5, "min_lon": 15, "max_lon": 30},
        # Southeast Asia (Indonesia, Malaysia)
        {"min_lat": -10, "max_lat": 10, "min_lon": 95, "max_lon": 140},
        # Northern US/Canada Forests
        {"min_lat": 45, "max_lat": 60, "min_lon": -125, "max_lon": -70},
        # European Forests
        {"min_lat": 45, "max_lat": 65, "min_lon": 5, "max_lon": 35},
        # Siberian Forests
        {"min_lat": 50, "max_lat": 65, "min_lon": 80, "max_lon": 130},
        # Brazilian Atlantic Forest
        {"min_lat": -30, "max_lat": -5, "min_lon": -55, "max_lon": -35},
        # US Eastern Forests
        {"min_lat": 30, "max_lat": 45, "min_lon": -90, "max_lon": -70},
        # Japan Forests
        {"min_lat": 30, "max_lat": 45, "min_lon": 130, "max_lon": 145},
        # Australian Forests
        {"min_lat": -40, "max_lat": -25, "min_lon": 140, "max_lon": 155}
    ]
    
    # Choose a random forest region
    region = random.choice(forest_regions)
    
    # Generate random coordinates within the chosen region
    lat = random.uniform(region["min_lat"], region["max_lat"])
    lon = random.uniform(region["min_lon"], region["max_lon"])
    
    return lat, lon

def get_random_land_coordinates():
    """Generate random coordinates that are likely to be on land, with preference for forested areas."""
    # 80% chance to pick from known forest areas
    if random.random() < 0.8:
        return get_forest_vegetation_regions()
    
    # 20% chance to pick from general continental areas
    continents = [
        # North America
        {"min_lat": 25, "max_lat": 70, "min_lon": -170, "max_lon": -50},
        # South America
        {"min_lat": -55, "max_lat": 15, "min_lon": -80, "max_lon": -35},
        # Europe
        {"min_lat": 35, "max_lat": 70, "min_lon": -10, "max_lon": 40},
        # Africa
        {"min_lat": -35, "max_lat": 35, "min_lon": -20, "max_lon": 55},
        # Asia
        {"min_lat": 0, "max_lat": 75, "min_lon": 40, "max_lon": 180},
        # Australia
        {"min_lat": -45, "max_lat": -10, "min_lon": 110, "max_lon": 155}
    ]
    
    # Choose a random continent
    continent = random.choice(continents)
    
    # Generate random coordinates within the chosen continent
    lat = random.uniform(continent["min_lat"], continent["max_lat"])
    lon = random.uniform(continent["min_lon"], continent["max_lon"])
    
    return lat, lon

def get_random_date(start_year=2015, end_year=2025):
    """Generate a random date between start_year and end_year."""
    # Limit to actual Sentinel-2 availability (launched June 2015)
    start_date = datetime(2015, 6, 23)
    end_date = min(datetime(end_year, 12, 31), datetime.now())
    
    # Calculate time difference in days
    time_diff = (end_date - start_date).days
    
    # Generate random days to add
    random_days = random.randint(0, time_diff)
    
    # Add random days to start date
    random_date = start_date + timedelta(days=random_days)
    
    return random_date.strftime("%Y-%m-%d")

def get_best_satellite_image(latitude, longitude, date_str, buffer_km=5, days_range=15, max_cloud_percent=20):
    """
    Fetch the best satellite image (with lowest cloud coverage) from Google Earth Engine.
    """
    print(f"Fetching best satellite image around {date_str} with cloud coverage <= {max_cloud_percent}%...")

    # Create a point geometry
    point = ee.Geometry.Point([longitude, latitude])

    # Create a buffer around the point
    region = point.buffer(buffer_km * 1000)

    # Parse the date
    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    
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
        
        # Add date difference from target date
        date_diff = ee.Number(img.get('system:time_start')).subtract(target_timestamp).abs()
        
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

def download_image(image_info, region, latitude, longitude, output_dir="datasets/iris/images"):
    """Download an image with natural color visualization and generate masks"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        masks_dir = os.path.join(os.path.dirname(output_dir), "masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        image = image_info['image']
        satellite = image_info['satellite']
        
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
            # Download the RGB image (easier to open than GeoTIFF)
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
                
            # Download the visualization of the mask
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

def check_vegetation_coverage(latitude, longitude):
    """
    Check if the coordinates have significant vegetation.
    Returns a tuple of (is_on_land, veg_score) where veg_score is 0-100.
    """
    try:
        # Create a point geometry
        point = ee.Geometry.Point([longitude, latitude])
        
        # Create a buffer around the point
        region = point.buffer(5000)  # 5km buffer
        
        # Use Sentinel-2 data to calculate NDVI
        # Take the last 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        s2_collection = (ee.ImageCollection("COPERNICUS/S2_SR")
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filterBounds(region)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
        
        if s2_collection.size().getInfo() == 0:
            # If no Sentinel-2 data, try Landsat
            l8_collection = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                .filterBounds(region))
            
            if l8_collection.size().getInfo() == 0:
                # No data available
                return (True, 50)  # Default to "maybe on land with average vegetation"
            
            # Get the most recent, least cloudy Landsat image
            l8_image = l8_collection.sort('CLOUD_COVER').first()
            
            # Calculate NDVI for Landsat
            ndvi = l8_image.normalizedDifference(['SR_B5', 'SR_B4'])
            
            # Check for water
            ndwi = l8_image.normalizedDifference(['SR_B3', 'SR_B5'])
            water_score = ndwi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=30,
                maxPixels=1e9
            ).get('nd')
            
            is_water = ee.Number(water_score).gt(0.1).getInfo()
            
        else:
            # Get the most recent, least cloudy Sentinel-2 image
            s2_image = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
            
            # Calculate NDVI for Sentinel-2
            ndvi = s2_image.normalizedDifference(['B8', 'B4'])
            
            # Check for water
            ndwi = s2_image.normalizedDifference(['B3', 'B8'])
            water_score = ndwi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=10,
                maxPixels=1e9
            ).get('nd')
            
            is_water = ee.Number(water_score).gt(0.1).getInfo()
        
        # Calculate mean NDVI
        mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).get('nd')
        
        # Convert NDVI (typically -1 to 1) to a 0-100 score
        veg_score = ((ee.Number(mean_ndvi).add(1)).divide(2).multiply(100)).getInfo()
        
        return (not is_water, veg_score)
    
    except Exception as e:
        print(f"Error checking vegetation coverage: {e}")
        # If we can't determine, assume it might be land with some vegetation
        return (True, 50)

def process_location(location_id, max_cloud_percent=20):
    """Process a single location and download the best image if available."""
    # Get random coordinates, preferring forested areas
    lat, lon = get_random_land_coordinates()
    
    print(f"\nLocation {location_id}: Checking coordinates ({lat:.4f}, {lon:.4f})...")
    
    # Check if on land and has vegetation
    is_land, veg_score = check_vegetation_coverage(lat, lon)
    
    if not is_land:
        print(f"  Coordinates appear to be on water. Skipping.")
        return False
    
    print(f"  Coordinates are on land with vegetation score {veg_score:.1f}/100")
    
    # Skip areas with low vegetation (below 40%)
    if veg_score < 40:
        print(f"  Vegetation score too low. Skipping.")
        return False
    
    # Get random date
    date_str = get_random_date()
    print(f"  Using date: {date_str}")
    
    try:
        # Get the best satellite image with cloud coverage below threshold
        image_info, region = get_best_satellite_image(
            lat, lon, date_str, 
            buffer_km=5,
            days_range=15,
            max_cloud_percent=max_cloud_percent
        )
        
        if image_info:
            # Download the image with natural colors
            success = download_image(image_info, region, lat, lon)
            return success
        
    except Exception as e:
        print(f"  Error processing location: {str(e)}")
    
    return False

def batch_image_download(num_locations=10, max_cloud_percent=20, max_workers=4):
    """Download images from multiple locations in parallel."""
    print(f"Starting batch satellite image download for {num_locations} locations...")
    print(f"Using {max_workers} parallel workers")
    print(f"Focusing on areas with higher vegetation and forest coverage")
    print(f"Images will be downloaded to datasets/iris/images and masks to datasets/iris/masks")
    print("Press Ctrl+C to stop the process.")
    
    # Initialize train.txt - clear it first
    with open("datasets/iris/train.txt", "w") as f:
        pass
    
    # Split data into train and validation sets
    val_ratio = 0.2  # 20% for validation
    val_count = int(num_locations * val_ratio)
    train_count = num_locations - val_count
    
    successful_downloads = 0
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_location, i+1, max_cloud_percent) for i in range(num_locations)]
            
            downloaded_files = []
            for i, future in enumerate(futures):
                if future.result():
                    successful_downloads += 1
                    # Store the filename (we would need to extract it from the function)
                    # For simplicity, we'll just use the counter
                    downloaded_files.append(i)
    
        # Now, let's split the data
        if successful_downloads > 0:
            random.shuffle(downloaded_files)
            val_files = downloaded_files[:val_count]
            
            # Move designated files to validation set
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
    
    print(f"\nFinal statistics: {successful_downloads} images downloaded from {num_locations} attempted locations.")

if __name__ == "__main__":
    # Specify how many locations to process
    num_locations = 5000
    # Number of parallel downloads
    max_workers = 4
    
    batch_image_download(num_locations=num_locations, max_cloud_percent=20, max_workers=max_workers)