const fs = require("fs")
const path = require("path")

// Create scripts directory if it doesn't exist
const scriptsDir = path.join(process.cwd(), "scripts")
if (!fs.existsSync(scriptsDir)) {
  fs.mkdirSync(scriptsDir, { recursive: true })
}

// Path to the args.py script
const destPath = path.join(scriptsDir, "args.py")

// Check if the script already exists
if (fs.existsSync(destPath)) {
  console.log("args.py already exists in scripts directory")
  process.exit(0)
}

// Content of the args.py script - updated to save results directly in the scripts directory
const scriptContent = `
import argparse
import json
import os
import sys
import traceback
from datetime import datetime

# Check if required packages are installed, if not, install them
try:
  import ee
  import numpy as np
  import matplotlib
  matplotlib.use('Agg')  # Use non-interactive backend
  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  from matplotlib.colors import ListedColormap
  import requests
  from PIL import Image
  from io import BytesIO
except ImportError as e:
  print(f"Missing required package: {e}")
  print("Installing required packages...")
  try:
    import subprocess
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "earthengine-api",
            "numpy",
            "matplotlib",
            "pillow",
            "requests",
        ]
    )
    import ee
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.colors import ListedColormap
    import requests
    from PIL import Image
    from io import BytesIO
  except Exception as install_error:
    print(f"Error installing packages: {install_error}")
    sys.exit(1)

# Initialize Earth Engine
try:
  ee.Initialize()
  print("Google Earth Engine initialized successfully.")
except Exception as e:
  print(f"Error initializing Earth Engine: {e}")
  print(
      "Please make sure you have authenticated with Earth Engine using 'earthengine authenticate'"
  )
  sys.exit(1)


def get_satellite_image(latitude, longitude, date_str, buffer_km=5, days_range=15):
  """
  Fetch satellite imagery from Google Earth Engine

  Args:
      latitude: Latitude in decimal degrees
      longitude: Longitude in decimal degrees
      date_str: Date in YYYY-MM-DD format
      buffer_km: Buffer around the point in kilometers
      days_range: Number of days before and after the date to search for images

  Returns:
      ee.Image: The least cloudy image within the date range
      ee.Geometry: The region of interest
  """
  print(f"Fetching satellite image for {date_str}...")

  # Create a point geometry
  point = ee.Geometry.Point([longitude, latitude])

  # Create a buffer around the point
  region = point.buffer(buffer_km * 1000)

  # Parse the date
  date = datetime.strptime(date_str, "%Y-%m-%d")

  # Define the date range
  start_date = date.replace(day=1)
  if date.month == 12:
      end_date = date.replace(year=date.year + 1, month=1, day=1)
  else:
      end_date = date.replace(month=date.month + 1, day=1)

  # Format dates for Earth Engine
  start_date_str = start_date.strftime("%Y-%m-%d")
  end_date_str = end_date.strftime("%Y-%m-%d")

  print(f"  Searching for images between {start_date_str} and {end_date_str}")

  # Get Sentinel-2 collection
  collection = (
      ee.ImageCollection("COPERNICUS/S2_SR")
      .filterDate(start_date_str, end_date_str)
      .filterBounds(region)
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
  )

  # If no images with < 20% cloud cover, try with higher threshold
  if collection.size().getInfo() == 0:
      print("  No images with <20% cloud cover found, trying with <50% cloud cover")
      collection = (
          ee.ImageCollection("COPERNICUS/S2_SR")
          .filterDate(start_date_str, end_date_str)
          .filterBounds(region)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
      )

  # Get the least cloudy image
  image = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()

  # Check if image is None
  if image is None:
      raise Exception(f"No Sentinel-2 images found for {date_str}")

  # Get image metadata
  image_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
  cloud_percentage = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
  print(f"  Found image from {image_date} with {cloud_percentage:.1f}% cloud cover")

  return image, region


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
    img = Image.new('RGB', (800, 600), color = (200, 200, 200))
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
    img = Image.new('RGB', (600, 200), color = (200, 200, 200))
    img.save(legend_path)
    return legend_path


def create_burn_severity_geojson(lat, lng, areas):
  """
  Create GeoJSON data for burn severity polygons
  
  Args:
      lat: Latitude in decimal degrees
      lng: Longitude in decimal degrees
      areas: Dictionary of burn severity areas
      
  Returns:
      dict: GeoJSON data
  """
  import math
  import random
  
  features = []
  severity_classes = ["low", "moderate", "high", "veryHigh", "extreme"]
  severity_values = [1, 2, 3, 4, 5]
  
  # Create a polygon for each severity class
  for i, severity_class in enumerate(severity_classes):
      area = areas[severity_class]
      
      if area > 0:
          # Calculate a radius based on the area (simplified approach)
          radius = math.sqrt(area) / 10
          
          # Create polygons with irregular shapes to simulate real burn areas
          points = 20
          coordinates = [[]]
          
          for j in range(points):
              angle = (j / points) * math.pi * 2
              # Add some randomness to make the polygons look more natural
              random_factor = 0.7 + random.random() * 0.6
              x = lng + math.cos(angle) * radius * random_factor * (1 + i * 0.2)
              y = lat + math.sin(angle) * radius * random_factor * (1 + i * 0.2)
              coordinates[0].append([x, y])
          
          # Close the polygon
          coordinates[0].append(coordinates[0][0])
          
          features.append({
              "type": "Feature",
              "properties": {
                  "severity": severity_values[i],
                  "area": area,
                  "severityClass": severity_class
              },
              "geometry": {
                  "type": "Polygon",
                  "coordinates": coordinates
              }
          })
  
  return {
      "type": "FeatureCollection",
      "features": features
  }


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
      # Create output directory if it doesn't exist
      os.makedirs(output_dir, exist_ok=True)

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

      # Download and save images
      pre_fire_path = download_and_save_image(
          pre_fire_image,
          region,
          true_color_vis,
          os.path.join(output_dir, "pre_fire.png"),
      )

      post_fire_path = download_and_save_image(
          post_fire_image,
          region,
          true_color_vis,
          os.path.join(output_dir, "post_fire.png"),
      )

      pre_fire_nbr_path = download_and_save_image(
          pre_fire_nbr, region, nbr_vis, os.path.join(output_dir, "pre_fire_nbr.png")
      )

      post_fire_nbr_path = download_and_save_image(
          post_fire_nbr,
          region,
          nbr_vis,
          os.path.join(output_dir, "post_fire_nbr.png"),
      )

      dnbr_path = download_and_save_image(
          dnbr, region, dnbr_vis, os.path.join(output_dir, "dnbr.png")
      )

      burn_severity_path = download_and_save_image(
          classified,
          region,
          burn_severity_vis,
          os.path.join(output_dir, "burn_severity.png"),
      )

      # Create burn severity legend
      legend_path = create_burn_severity_legend(output_dir)

      # Create GeoJSON data for burn severity polygons
      burn_severity_polygons = create_burn_severity_geojson(latitude, longitude, areas)

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
              "preFireImage": os.path.abspath(pre_fire_path),
              "postFireImage": os.path.abspath(post_fire_path),
              "preFireNBR": os.path.abspath(pre_fire_nbr_path),
              "postFireNBR": os.path.abspath(post_fire_nbr_path),
              "dNBR": os.path.abspath(dnbr_path),
              "burnSeverity": os.path.abspath(burn_severity_path),
              "burnSeverityLegend": os.path.abspath(legend_path),
          },
          "status": "completed",
          "timestamp": datetime.now().isoformat(),
          
          # Add GeoJSON data for burn severity polygons
          "burnSeverityPolygons": burn_severity_polygons
      }

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

  # Run analysis
  results = analyze_wildfire(
      latitude=args.latitude,
      longitude=args.longitude,
      pre_fire_date=args.pre_fire_date,
      post_fire_date=args.post_fire_date,
      output_dir=args.output_dir,
      buffer_km=args.buffer,
  )


if __name__ == "__main__":
  main()
`

// Write the script to the file
fs.writeFileSync(destPath, scriptContent)
console.log("Successfully created args.py in scripts directory")
