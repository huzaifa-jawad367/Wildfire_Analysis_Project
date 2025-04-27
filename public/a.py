import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
from skimage.measure import find_contours

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        "earthengine-api": "ee",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "pillow": "PIL",
        "requests": "requests",
        "scikit-image": "skimage",
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing required packages...")
        try:
            import subprocess
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                *missing_packages
            ])
            print("Successfully installed required packages.")
        except Exception as install_error:
            error_msg = f"Error installing packages: {str(install_error)}"
            print(error_msg)
            save_error_json(error_msg)
            sys.exit(1)

def save_error_json(error_message, output_dir=None, details=None):
    """Save error information to error.json"""
    error_data = {
        "error": str(error_message),
        "details": details or {},
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        if output_dir:
            error_path = os.path.join(output_dir, "error.json")
            os.makedirs(output_dir, exist_ok=True)
            with open(error_path, "w") as f:
                json.dump(error_data, f, indent=2)
    except Exception as e:
        print(f"Error saving error.json: {e}")

# Check dependencies first
check_dependencies()

# Now import the required packages
try:
    import ee
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.colors import ListedColormap
    import requests
    from PIL import Image
    from io import BytesIO
    from skimage.measure import find_contours
except Exception as e:
    error_msg = f"Error importing required packages: {str(e)}"
    print(error_msg)
    save_error_json(error_msg)
        sys.exit(1)

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
    save_error_json(error_msg)
    sys.exit(1)


def get_satellite_image(latitude, longitude, date_str, buffer_km=5, days_range=15):
    """
    Fetch satellite imagery from Google Earth Engine closest to the specified date,
    prioritizing images with minimal cloud cover in the region of interest.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        date_str: Date in YYYY-MM-DD format
        buffer_km: Buffer around the point in kilometers
        days_range: Number of days before and after the date to search for images

    Returns:
        ee.Image: The best available image (considering date and cloud cover)
        ee.Geometry: The region of interest
    """
    print(f"Fetching satellite image closest to {date_str}...")

    # Create a point geometry
    point = ee.Geometry.Point([longitude, latitude])

    # Create a buffer around the point
    region = point.buffer(buffer_km * 1000)

    # Parse the date
    target_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Define the date range
    start_date = target_date - timedelta(days=days_range)
    end_date = target_date + timedelta(days=days_range)

    # Format dates for Earth Engine
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    print(f"  Searching for images between {start_date_str} and {end_date_str}")

    # Get Sentinel-2 collection
    collection = (ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start_date_str, end_date_str)
                 .filterBounds(region))

    # Convert target date to milliseconds since epoch for Earth Engine
    target_timestamp = int(target_date.timestamp() * 1000)

    def add_cloud_score(img):
        # Calculate cloud score for the region of interest
        cloud_score = img.select(['MSK_CLDPRB']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,  # Scale in meters
            maxPixels=1e9
        ).get('MSK_CLDPRB')
        
        # Add cloud score and date difference as properties
        date_diff = ee.Number(img.get('system:time_start')).subtract(target_timestamp).abs()
        return img.set({
            'cloud_score': cloud_score,
            'date_diff': date_diff
        })

    # Add cloud scores to all images
    collection_with_scores = collection.map(add_cloud_score)

    # Get all images as a list
    image_list = collection_with_scores.toList(collection_with_scores.size())

    # Function to find best image
    def find_best_image():
        images_info = []
        size = image_list.size().getInfo()
        
        print(f"  Found {size} total images")
        
        for i in range(size):
            try:
                img = ee.Image(image_list.get(i))
                cloud_score = img.get('cloud_score').getInfo()
                date_diff = img.get('date_diff').getInfo()
                date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                
                images_info.append({
                    'index': i,
                    'cloud_score': cloud_score,
                    'date_diff': date_diff,
                    'date': date
                })
                
                print(f"    Image {i+1}: Date={date}, Cloud Score={cloud_score:.1f}%, Time Diff={date_diff/(1000*60*60*24):.1f} days")
                
            except Exception as e:
                print(f"    Error processing image {i+1}: {e}")
                continue
        
        if not images_info:
            raise Exception("No valid images found")
        
        # Sort by cloud score first, then by date difference
        images_info.sort(key=lambda x: (x['cloud_score'], x['date_diff']))
        
        best_image = images_info[0]
        print(f"\n  Selected best image:")
        print(f"    Date: {best_image['date']}")
        print(f"    Cloud coverage: {best_image['cloud_score']:.1f}%")
        print(f"    Time difference: {best_image['date_diff']/(1000*60*60*24):.1f} days")
        
        return ee.Image(image_list.get(best_image['index']))

    try:
        best_image = find_best_image()
        return best_image, region
        
    except Exception as e:
        print(f"Error finding best image: {e}")
        raise Exception(
            f"No suitable Sentinel-2 images found for {date_str} (±{days_range} days)"
        )


def calculate_nbr(image):
    """
    Calculate Normalized Burn Ratio (NBR) for a Sentinel-2 image

    Args:
        image: Sentinel-2 image

    Returns:
        ee.Image: NBR image
    """
    # For Sentinel-2: NIR = B8, SWIR = B12
    return image.normalizedDifference(["B8", "B12"]).rename("NBR")


def calculate_dnbr(pre_fire_nbr, post_fire_nbr):
    """
    Calculate difference in NBR (dNBR) between pre-fire and post-fire images

    Args:
        pre_fire_nbr: Pre-fire NBR image
        post_fire_nbr: Post-fire NBR image

    Returns:
        ee.Image: dNBR image
    """
    return pre_fire_nbr.subtract(post_fire_nbr).rename("dNBR")


def classify_burn_severity(dnbr):
    """
    Classify burn severity based on dNBR values

    Args:
        dnbr: dNBR image

    Returns:
        ee.Image: Classified burn severity image
    """
    # Define thresholds for burn severity classes
    thresholds = [0.05, 0.1, 0.2, 0.3]
    class_values = [1, 2, 3, 4, 5]  # 1=low, 2=moderate, 3=high, 4=very high, 5=extreme

    # Create a classified image
    classified = ee.Image(0)

    # Add each class
    classified = classified.where(
        dnbr.gt(0).And(dnbr.lte(thresholds[0])), class_values[0]
    )
    classified = classified.where(
        dnbr.gt(thresholds[0]).And(dnbr.lte(thresholds[1])), class_values[1]
    )
    classified = classified.where(
        dnbr.gt(thresholds[1]).And(dnbr.lte(thresholds[2])), class_values[2]
    )
    classified = classified.where(
        dnbr.gt(thresholds[2]).And(dnbr.lte(thresholds[3])), class_values[3]
    )
    classified = classified.where(dnbr.gt(thresholds[3]), class_values[4])

    return classified.rename("burn_severity")


def calculate_area_by_severity(classified, region, scale=30):
    """
    Calculate area by burn severity class

    Args:
        classified: Classified burn severity image
        region: Region to calculate area for
        scale: Scale in meters

    Returns:
        dict: Area by severity class in square kilometers
    """
    print("Calculating burn area by severity class...")

    # Define class names
    class_names = ["low", "moderate", "high", "veryHigh", "extreme"]

    # Calculate area for each class
    areas = {}
    for i, name in enumerate(class_names):
        class_value = i + 1
        area_image = classified.eq(class_value)
        area_m2 = (
            area_image.multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(), geometry=region, scale=scale, maxPixels=1e9
            )
            .get("burn_severity")
        )

        # Convert to square kilometers
        areas[name] = ee.Number(area_m2).divide(1e6).getInfo()
        print(f"  {name.capitalize()} severity: {areas[name]:.2f} km²")

    return areas


def get_nbr_stats(pre_fire_nbr, post_fire_nbr, dnbr, region, scale=30):
    """
    Get NBR statistics

    Args:
        pre_fire_nbr: Pre-fire NBR image
        post_fire_nbr: Post-fire NBR image
        dnbr: dNBR image
        region: Region to calculate statistics for
        scale: Scale in meters

    Returns:
        dict: NBR statistics
    """
    print("Calculating NBR statistics...")

    # Calculate statistics for pre-fire NBR
    pre_fire_stats = pre_fire_nbr.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, maxPixels=1e9
    ).getInfo()

    # Calculate statistics for post-fire NBR
    post_fire_stats = post_fire_nbr.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale, maxPixels=1e9
    ).getInfo()

    # Calculate statistics for dNBR
    dnbr_stats = dnbr.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.max(), sharedInputs=True),
        geometry=region,
        scale=scale,
        maxPixels=1e9,
    ).getInfo()

    stats = {
        "preFireAvg": pre_fire_stats["NBR"],
        "postFireAvg": post_fire_stats["NBR"],
        "dNBRAvg": dnbr_stats["dNBR_mean"],
        "dNBRMax": dnbr_stats["dNBR_max"],
    }

    print(f"  Pre-fire NBR (avg): {stats['preFireAvg']:.3f}")
    print(f"  Post-fire NBR (avg): {stats['postFireAvg']:.3f}")
    print(f"  dNBR (avg): {stats['dNBRAvg']:.3f}")
    print(f"  dNBR (max): {stats['dNBRMax']:.3f}")

    return stats


def download_and_save_image(image, region, vis_params, filename, scale=30):
    """
    Download an image from Earth Engine and save it to a file

    Args:
        image: Earth Engine image
        region: Region to download
        vis_params: Visualization parameters
        filename: Output filename
        scale: Scale in meters

    Returns:
        str: Path to the saved image
    """
    print(f"Downloading and saving {filename}...")

    try:
        # Get the image URL
        url = image.getThumbURL(
            {
                "region": region,
                "dimensions": "800x600",
                "format": "png",
                "min": vis_params.get("min", 0),
                "max": vis_params.get("max", 1),
                "bands": vis_params.get("bands", None),
                "palette": vis_params.get("palette", None),
            }
        )

        # Download the image
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # Save the image
        img.save(filename)
        print(f"  Saved to {filename}")

        return filename
    except Exception as e:
        print(f"Error downloading image: {e}")
        # Create a placeholder image
        img = Image.new("RGB", (800, 600), color=(200, 200, 200))
        img.save(filename)
        print(f"  Created placeholder image at {filename}")
        return filename


def create_burn_severity_legend(output_dir):
    """
    Create a legend for the burn severity map

    Args:
        output_dir: Output directory

    Returns:
        str: Path to the legend image
    """
    print("Creating burn severity legend...")

    try:
        # Create figure and axis
        fig, ax = plt.figure(figsize=(6, 2)), plt.gca()

        # Define colors and labels
        colors = ["#69B34C", "#FAB733", "#FF8E15", "#FF4E11", "#FF0D0D"]
        labels = ["Low", "Moderate", "High", "Very High", "Extreme"]

        # Create color patches
        for i, (color, label) in enumerate(zip(colors, labels)):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
            ax.text(i + 0.5, 0.5, label, ha="center", va="center")

        # Set axis limits and remove ticks
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Set title
        ax.set_title("Burn Severity Legend")

        # Save figure
        legend_path = os.path.join(output_dir, "burn_severity_legend.png")
        plt.savefig(legend_path, bbox_inches="tight")
        plt.close()

        print(f"  Saved to {legend_path}")

        return legend_path
    except Exception as e:
        print(f"Error creating legend: {e}")
        # Return a placeholder path
        legend_path = os.path.join(output_dir, "burn_severity_legend.png")
        # Create a simple placeholder image
        img = Image.new("RGB", (600, 200), color=(200, 200, 200))
        img.save(legend_path)
        return legend_path


def extract_high_severity_coordinates(image_path, region):
    """
    Extract coordinates of extreme severity (red) burn areas from the burn severity image.
    Focuses on the top 15% most concentrated red areas to identify the most severely burned regions.
    Returns empty list if no areas meet the threshold criteria.
    """
    try:
        print(f"Starting coordinate extraction from {image_path}")
        
        # Load the image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        print(f"Image loaded with shape: {img_array.shape}")
        
        # Define the RGB values for red (extreme severity)
        red_color = np.array([255, 13, 13])    # Extreme severity (red)
        
        # Create mask for red pixels with more tolerance
        def color_mask(color, tolerance=40):  # Increased tolerance from 30 to 40
            return np.all(np.abs(img_array - color) < tolerance, axis=2)
        
        red_mask = color_mask(red_color)
        
        # Check if there are any red pixels at all
        if not np.any(red_mask):
            print("No red pixels found in the image")
            return []
        
        # Get region bounds
        try:
            bounds = region.bounds().getInfo()
            print(f"Raw bounds data: {bounds}")
            
            # Handle different bounds formats
            if isinstance(bounds, list):
                min_lon, min_lat, max_lon, max_lat = bounds
            elif isinstance(bounds, dict) and 'coordinates' in bounds:
                coords = bounds['coordinates'][0]
                min_lon = min(c[0] for c in coords)
                max_lon = max(c[0] for c in coords)
                min_lat = min(c[1] for c in coords)
                max_lat = max(c[1] for c in coords)
            else:
                raise ValueError(f"Unexpected bounds format: {bounds}")
            
            print(f"Processed bounds: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
            
            # Get image dimensions
            height, width = img_array.shape[:2]
            
            # Process image in 16x16 blocks to calculate concentration
            block_size = 16
            block_concentrations = []
            block_positions = []
            
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    # Get the block
                    y_end = min(y + block_size, height)
                    x_end = min(x + block_size, width)
                    block_red = red_mask[y:y_end, x:x_end]
                    
                    # Calculate concentration of red pixels in this block
                    concentration = np.sum(block_red) / ((y_end - y) * (x_end - x))
                    
                    if concentration > 0:
                        block_concentrations.append(concentration)
                        block_positions.append((x, y, x_end, y_end))
            
            if not block_concentrations:
                print("No blocks with red pixels found")
                return []
            
            # Find threshold for top 15% most concentrated blocks (relaxed from 5%)
            concentration_threshold = np.percentile(block_concentrations, 85)
            
            # Additional minimum threshold check (relaxed from 0.3 to 0.2)
            min_acceptable_concentration = 0.2  # At least 20% of pixels should be red
            final_threshold = max(concentration_threshold, min_acceptable_concentration)
            
            coordinates = []
            
            # Process only blocks with high concentration
            for concentration, (x, y, x_end, y_end) in zip(block_concentrations, block_positions):
                if concentration >= final_threshold:
                    # Use center of block for coordinate
                    block_center_x = x + (x_end - x) / 2
                    block_center_y = y + (y_end - y) / 2
                    
                    # Convert to normalized coordinates (0-1)
                    norm_x = block_center_x / width
                    norm_y = 1 - (block_center_y / height)  # Flip Y coordinate
                    
                    # Convert to geographic coordinates
                    lon = min_lon + (max_lon - min_lon) * norm_x
                    lat = min_lat + (max_lat - min_lat) * norm_y
                    
                    coordinates.append({
                        'latitude': lat,
                        'longitude': lon,
                        'severity': 'extreme',
                        'concentration': float(concentration)  # Add concentration info for debugging
                    })
            
            if not coordinates:
                print("No areas meet the concentration threshold criteria")
                return []
            
            print(f"Extracted {len(coordinates)} coordinates from top 15% most concentrated areas")
            print(f"Concentration threshold: {final_threshold:.3f}")
            if coordinates:
                print(f"Sample coordinate with concentration: {coordinates[0]}")
                print(f"Number of high severity points identified: {len(coordinates)}")
            
            return coordinates
            
        except Exception as bounds_error:
            print(f"Error processing bounds: {bounds_error}")
            traceback.print_exc()
            return []
        
    except Exception as e:
        print(f"Error extracting high severity coordinates: {e}")
        traceback.print_exc()
        return []


def analyze_wildfire(
    latitude, longitude, pre_fire_date, post_fire_date, output_dir, buffer_km=5
):
    """
    Analyze wildfire damage using satellite imagery

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        pre_fire_date: Pre-fire date in YYYY-MM-DD format
        post_fire_date: Post-fire date in YYYY-MM-DD format
        output_dir: Output directory
        buffer_km: Buffer around the point in kilometers

    Returns:
        dict: Analysis results
    """
    try:
        # Create output directory and images subdirectory if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        print("=== Starting Wildfire Damage Analysis ===")
        print(f"Location: {latitude}, {longitude}")
        print(f"Pre-fire date: {pre_fire_date}")
        print(f"Post-fire date: {post_fire_date}")
        print(f"Buffer: {buffer_km} km")
        print(f"Output directory: {output_dir}")
        print("=======================================")

        # Get pre-fire and post-fire images
        pre_fire_image, region = get_satellite_image(
            latitude, longitude, pre_fire_date, buffer_km
        )
        post_fire_image, _ = get_satellite_image(
            latitude, longitude, post_fire_date, buffer_km
        )

        print("Calculating burn indices...")
        # Calculate NBR for pre-fire and post-fire images
        pre_fire_nbr = calculate_nbr(pre_fire_image)
        post_fire_nbr = calculate_nbr(post_fire_image)

        # Calculate dNBR
        dnbr = calculate_dnbr(pre_fire_nbr, post_fire_nbr)

        # Classify burn severity
        classified = classify_burn_severity(dnbr)

        # Calculate area by severity class
        print("Calculating burn statistics...")
        areas = calculate_area_by_severity(classified, region)

        # Get NBR statistics
        nbr_stats = get_nbr_stats(pre_fire_nbr, post_fire_nbr, dnbr, region)

        # Calculate total burned area
        total_burned_area = sum(areas.values())
        print(f"Total burned area: {total_burned_area:.2f} km²")

        # Download images
        print("Downloading satellite images and maps...")

        # True color visualization
        true_color_vis = {"min": 0, "max": 3000, "bands": ["B4", "B3", "B2"]}

        # NBR visualization
        nbr_vis = {"min": -1, "max": 1, "palette": ["#FF0000", "#FFFFFF", "#00FF00"]}

        # dNBR visualization
        dnbr_vis = {
            "min": -0.5,
            "max": 0.5,
            "palette": ["#00FF00", "#FFFFFF", "#FF0000"],
        }

        # Burn severity visualization
        burn_severity_vis = {
            "min": 1,
            "max": 5,
            "palette": ["#69B34C", "#FAB733", "#FF8E15", "#FF4E11", "#FF0D0D"],
        }

        # Download and save true color images (closest to target dates)
        pre_fire_path = download_and_save_image(
            pre_fire_image,
            region,
            true_color_vis,
            os.path.join(images_dir, "pre_fire.png"),
        )

        post_fire_path = download_and_save_image(
            post_fire_image,
            region,
            true_color_vis,
            os.path.join(images_dir, "post_fire.png"),
        )

        # Download and save other images
        pre_fire_nbr_path = download_and_save_image(
            pre_fire_nbr, region, nbr_vis, os.path.join(images_dir, "pre_fire_nbr.png")
        )

        post_fire_nbr_path = download_and_save_image(
            post_fire_nbr,
            region,
            nbr_vis,
            os.path.join(images_dir, "post_fire_nbr.png"),
        )

        dnbr_path = download_and_save_image(
            dnbr, region, dnbr_vis, os.path.join(images_dir, "dnbr.png")
        )

        burn_severity_path = download_and_save_image(
            classified,
            region,
            burn_severity_vis,
            os.path.join(images_dir, "burn_severity.png"),
        )

        # Create burn severity legend
        legend_path = create_burn_severity_legend(images_dir)

        # Extract high severity coordinates from burn severity image
        print("\nExtracting high severity coordinates...")
        high_severity_coords = extract_high_severity_coordinates(burn_severity_path, region)
        print(f"Number of high severity coordinates: {len(high_severity_coords)}")

        # Save results to JSON
        results = {
            "location": {"latitude": latitude, "longitude": longitude},
            "preFireDate": pre_fire_date,
            "postFireDate": post_fire_date,
            "dataSource": "Sentinel-2",
            "totalBurnedArea": total_burned_area,
            "burnSeverityStats": areas,
            "nbrStats": nbr_stats,
            "images": {
                "preFireTrueColor": "images/pre_fire.png",
                "postFireTrueColor": "images/post_fire.png",
                "preFireNBR": "images/pre_fire_nbr.png",
                "postFireNBR": "images/post_fire_nbr.png",
                "dNBR": "images/dnbr.png",
                "burnSeverity": "images/burn_severity.png",
                "burnSeverityLegend": "images/burn_severity_legend.png",
            },
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "highSeverityCoordinates": high_severity_coords,
        }

        # Print a sample of the results for debugging
        print("\nResults preview:")
        print(f"Total burned area: {total_burned_area:.2f} km²")
        print(f"Number of high severity coordinates: {len(high_severity_coords)}")
        if high_severity_coords:
            print(f"Sample coordinate: {high_severity_coords[0]}")

        # Save results to JSON file
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_path}")
        print("=== Analysis completed successfully ===")

        return results

    except Exception as e:
        print(f"Error analyzing wildfire: {e}")
        print("Traceback:")
        traceback.print_exc()

        # Save error to JSON file
        error_results = {
            "location": {"latitude": latitude, "longitude": longitude},
            "preFireDate": pre_fire_date,
            "postFireDate": post_fire_date,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save error results to JSON file
        error_path = os.path.join(output_dir, "error.json")
        with open(error_path, "w") as f:
            json.dump(error_results, f, indent=2)

        print(f"Error details saved to {error_path}")
        print("=== Analysis failed ===")

        return error_results


def main():
    """
    Main function to run the wildfire analysis from command line
    """
    parser = argparse.ArgumentParser(
        description="Analyze wildfire damage using satellite imagery"
    )
    parser.add_argument(
        "--latitude", type=float, required=True, help="Latitude in decimal degrees"
    )
    parser.add_argument(
        "--longitude", type=float, required=True, help="Longitude in decimal degrees"
    )
    parser.add_argument(
        "--pre-fire-date",
        type=str,
        required=True,
        help="Pre-fire date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--post-fire-date",
        type=str,
        required=True,
        help="Post-fire date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=5.0,
        help="Buffer around the point in kilometers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory",
    )

    args = parser.parse_args()

    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create images subdirectory
        images_dir = os.path.join(args.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    # Run analysis
    results = analyze_wildfire(
        latitude=args.latitude,
        longitude=args.longitude,
        pre_fire_date=args.pre_fire_date,
        post_fire_date=args.post_fire_date,
        output_dir=args.output_dir,
        buffer_km=args.buffer,
    )

        # Print success message
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {os.path.join(args.output_dir, 'results.json')}")
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        print(f"\nError: {error_msg}")
        print("\nTraceback:")
        traceback.print_exc()
        
        # Save error information
        save_error_json(
            error_msg,
            args.output_dir,
            {
                "traceback": traceback.format_exc(),
                "parameters": {
                    "latitude": args.latitude,
                    "longitude": args.longitude,
                    "pre_fire_date": args.pre_fire_date,
                    "post_fire_date": args.post_fire_date,
                    "buffer": args.buffer,
                    "output_dir": args.output_dir,
                }
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
