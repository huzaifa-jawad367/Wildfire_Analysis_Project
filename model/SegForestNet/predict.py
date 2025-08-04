import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation
from dataset import MultiClassLandCoverDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from skimage.segmentation import slic
import torch.nn as nn
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Define color map for visualization
# Updated with more distinct colors for better visualization
def get_color_map():
    return {
        0: [0, 0, 0],       # Background - Black
        1: [0, 0, 255],     # Water - Blue
        2: [165, 42, 42],   # Terrain - Brown
        3: [0, 255, 0],     # Vegetation - Bright Green
        4: [0, 100, 0],     # Forest - Dark Green
        5: [255, 255, 255]  # Cloud - White
    }

# Class map for labels
CLASS_MAP = {
    0: "Background",
    1: "Water",
    2: "Terrain",
    3: "Vegetation",
    4: "Forest",
    5: "Cloud"
}

def post_process_prediction(prediction, original_image, n_segments=100):
    """Apply SLIC segmentation to smooth predictions."""
    # Convert prediction to numpy
    if isinstance(prediction, torch.Tensor):
        pred_np = prediction.cpu().numpy()
    else:
        pred_np = prediction
        
    # Convert image to numpy for segmentation
    if isinstance(original_image, torch.Tensor):
        # Take only the RGB channels if there are more
        if original_image.shape[0] > 3:
            image_np = original_image[:3].cpu().numpy().transpose(1, 2, 0)
        else:
            image_np = original_image.cpu().numpy().transpose(1, 2, 0)
    else:
        image_np = original_image
    
    # Ensure image is in correct range for SLIC
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    # Apply SLIC segmentation
    segments = slic(image_np, n_segments=n_segments, compactness=10, sigma=1, 
                   start_label=0, channel_axis=2)
    
    # Create smoothed prediction
    if len(pred_np.shape) > 2:  # Multi-class one-hot predictions
        smoothed_pred = np.zeros_like(pred_np)
        
        # For each segment, use majority vote for class
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            # Expand mask to match prediction shape
            expanded_mask = np.expand_dims(mask, axis=0)
            if len(pred_np.shape) == 3:
                expanded_mask = np.repeat(expanded_mask, pred_np.shape[0], axis=0)
            
            # Get majority class in this segment
            segment_pixels = pred_np[:, mask]
            if segment_pixels.size > 0:
                class_counts = np.sum(segment_pixels, axis=1)
                majority_class = np.argmax(class_counts)
                
                # Set the segment to the majority class
                smoothed_pred[:, mask] = 0
                smoothed_pred[majority_class, mask] = 1
    else:  # Binary prediction
        smoothed_pred = np.zeros_like(pred_np)
        
        # For each segment, use majority vote
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            votes = pred_np[mask]
            if votes.size > 0:
                majority = np.mean(votes) > 0.5
                smoothed_pred[mask] = majority
    
    # Convert back to tensor if input was tensor
    if isinstance(prediction, torch.Tensor):
        device = prediction.device
        smoothed_pred = torch.from_numpy(smoothed_pred).to(device)
        
    return smoothed_pred

def create_colored_mask(mask, color_map=None):
    """Create a colored mask from a class mask"""
    if color_map is None:
        color_map = get_color_map()
    
    # Create a colored mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Apply colors based on class indices
    for class_idx, color in color_map.items():
        colored_mask[mask == class_idx] = color
        
    return colored_mask

def predict_mask(model, image_path, transform, device, num_classes=6, img_size=512, use_indices=True, is_sentinel=False):
    """Predict mask for a single image"""
    # Load image
    dataset = MultiClassLandCoverDataset(
        image_dir=os.path.dirname(image_path),
        mask_dir=None,
        file_list=[os.path.basename(image_path)],
        transform=transform,
        img_size=img_size,
        inference=True,
        num_classes=num_classes,
        use_indices=use_indices,
        is_sentinel=is_sentinel
    )
    
    # Get the image
    image_data = dataset[0]
    image_tensor = image_data["image"].unsqueeze(0).to(device)  # Add batch dimension
    
    print(f"Input image tensor shape: {image_tensor.shape}")
    print(f"Input tensor min: {image_tensor.min()}, max: {image_tensor.max()}")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs.logits
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Print raw output shape
        print(f"Raw output shape: {logits.shape}")
        print(f"Raw output min: {logits.min().item()}, max: {logits.max().item()}")
        
        # Post-process prediction
        pred_mask = post_process_prediction(logits)
        pred_mask = pred_mask[0].cpu().numpy()  # Remove batch dimension
        
        # Get class distribution
        unique_classes, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size
        print("\nPredicted Class Distribution:")
        for cls_idx, count in zip(unique_classes, counts):
            percentage = (count / total_pixels) * 100
            if cls_idx in CLASS_MAP:
                class_name = CLASS_MAP[cls_idx]
            else:
                class_name = f"Unknown Class {cls_idx}"
            print(f"  {class_name}: {count} pixels ({percentage:.2f}%)")
    
    # Return the prediction and the original image
    return pred_mask, image_data["filename"]

def predict_directory(model, input_dir, output_dir, transform, device, num_classes=6, img_size=512, use_indices=True, is_sentinel=False):
    """Predict masks for all images in a directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.TIF'))]
    
    for image_file in tqdm(image_files, desc="Predicting masks"):
        image_path = os.path.join(input_dir, image_file)
        
        # Predict mask
        pred_mask, _ = predict_mask(model, image_path, transform, device, num_classes, img_size, use_indices, is_sentinel)
        
        # Create colored mask
        colored_mask = create_colored_mask(pred_mask)
        
        # Save masks
        image_name = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_mask.png"), pred_mask)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_mask_colored.png"), colored_mask)
        
        # Create visualization
        original_image = cv2.imread(image_path)
        original_image = cv2.resize(original_image, (pred_mask.shape[1], pred_mask.shape[0]))
        
        # Create a visualization by blending the original image with the colored mask
        alpha = 0.5
        viz = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
        
        # Save visualization
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_viz.png"), viz)
    
    print(f"Predictions saved to {output_dir}")

def check_and_load_ground_truth(img_filename, mask_dir):
    """Try to find and load the ground truth mask for comparison"""
    if mask_dir is None or not os.path.exists(mask_dir):
        return None, None
    
    # Try different mask naming conventions
    mask_candidates = [
        f"{os.path.splitext(img_filename)[0]}_mask.png",  # Standard naming convention
        f"{os.path.splitext(img_filename)[0]}_mask.jpg",
        f"{os.path.splitext(img_filename)[0]}_mask.tif",
        img_filename  # Same filename as image
    ]
    
    for mask_file in mask_candidates:
        mask_path = os.path.join(mask_dir, mask_file)
        if os.path.exists(mask_path):
            try:
                mask = np.array(Image.open(mask_path))
                
                # If mask has 3 channels (RGB), convert to class indices
                if len(mask.shape) == 3 and mask.shape[2] > 1:
                    # Create a properly indexed mask based on colors
                    new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                    
                    # Background/Black
                    black_mask = (mask[:,:,0] < 30) & (mask[:,:,1] < 30) & (mask[:,:,2] < 30)
                    new_mask[black_mask] = 0  # Background class
                    
                    # Water/Blue
                    blue_mask = (mask[:,:,0] < 30) & (mask[:,:,1] < 30) & (mask[:,:,2] > 200)
                    new_mask[blue_mask] = 1  # Water class
                    
                    # Terrain/Brown
                    brown_mask = (mask[:,:,0] > 100) & (mask[:,:,0] < 200) & (mask[:,:,1] < 100) & (mask[:,:,2] < 100)
                    new_mask[brown_mask] = 2  # Terrain class
                    
                    # Vegetation/Bright Green
                    bright_green_mask = (mask[:,:,0] < 30) & (mask[:,:,1] > 200) & (mask[:,:,2] < 30)
                    new_mask[bright_green_mask] = 3  # Vegetation class
                    
                    # Forest/Dark Green
                    dark_green_mask = (mask[:,:,0] < 30) & (mask[:,:,1] > 30) & (mask[:,:,1] < 150) & (mask[:,:,2] < 30)
                    new_mask[dark_green_mask] = 4  # Forest class
                    
                    # Cloud/White
                    white_mask = (mask[:,:,0] > 200) & (mask[:,:,1] > 200) & (mask[:,:,2] > 200)
                    new_mask[white_mask] = 5  # Cloud class
                    
                    mask = new_mask
                
                # Ensure correct class mapping for multi-class segmentation
                mask = np.clip(mask, 0, 5).astype(np.uint8)
                
                print(f"Found ground truth mask: {mask_path}")
                return mask, mask_path
            
            except Exception as e:
                print(f"Error loading mask {mask_path}: {str(e)}")
    
    print("No ground truth mask found for comparison.")
    return None, None

def create_comparison_visualization(original_img, pred_mask, gt_mask=None, output_path=None):
    """Create a comprehensive visualization comparing original, prediction, and ground truth"""
    color_map = get_color_map()
    
    # Create figure with subplots
    if gt_mask is not None:
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert original image to RGB if in BGR (OpenCV)
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        if original_img.dtype == np.float32:
            original_img = (original_img * 255).astype(np.uint8)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        original_img_rgb = original_img
    
    # Plot original image
    if gt_mask is not None:
        axs[0, 0].imshow(original_img_rgb)
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
    else:
        axs[0].imshow(original_img_rgb)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
    
    # Create colored prediction mask
    pred_colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        pred_colored[pred_mask == class_idx] = color
    
    # Plot prediction
    if gt_mask is not None:
        axs[0, 1].imshow(pred_colored)
        axs[0, 1].set_title('Predicted Classes')
        axs[0, 1].axis('off')
    else:
        axs[1].imshow(pred_colored)
        axs[1].set_title('Predicted Classes')
        axs[1].axis('off')
    
    # Create overlay of prediction on original image
    alpha = 0.5
    overlay = original_img_rgb.copy().astype(np.float32)
    for class_idx, color in color_map.items():
        mask = pred_mask == class_idx
        for c in range(3):
            overlay[:, :, c] = np.where(mask, 
                                        overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                                        overlay[:, :, c])
    overlay = overlay.astype(np.uint8)
    
    # Plot overlay
    if gt_mask is not None:
        axs[0, 2].imshow(overlay)
        axs[0, 2].set_title('Prediction Overlay')
        axs[0, 2].axis('off')
    else:
        axs[2].imshow(overlay)
        axs[2].set_title('Prediction Overlay')
        axs[2].axis('off')
    
    # If ground truth is available, add second row
    if gt_mask is not None:
        # Create colored ground truth mask
        gt_colored = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
        for class_idx, color in color_map.items():
            gt_colored[gt_mask == class_idx] = color
        
        # Plot ground truth
        axs[1, 0].imshow(gt_colored)
        axs[1, 0].set_title('Ground Truth')
        axs[1, 0].axis('off')
        
        # Create difference mask (where prediction differs from ground truth)
        diff_mask = pred_mask != gt_mask
        diff_image = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        diff_image[diff_mask] = [255, 0, 0]  # Red for differences
        
        # Plot difference
        axs[1, 1].imshow(diff_image)
        axs[1, 1].set_title('Prediction Errors (Red)')
        axs[1, 1].axis('off')
        
        # Create pie chart of class distribution
        predicted_classes, pred_counts = np.unique(pred_mask, return_counts=True)
        gt_classes, gt_counts = np.unique(gt_mask, return_counts=True)
        
        # Combine for complete comparison
        all_classes = np.unique(np.concatenate((predicted_classes, gt_classes)))
        
        # Prepare arrays for plotting
        pred_dist = np.zeros(len(all_classes))
        gt_dist = np.zeros(len(all_classes))
        class_names = []
        colors = []
        
        for i, cls in enumerate(all_classes):
            idx_pred = np.where(predicted_classes == cls)[0]
            idx_gt = np.where(gt_classes == cls)[0]
            
            pred_dist[i] = pred_counts[idx_pred[0]] if len(idx_pred) > 0 else 0
            gt_dist[i] = gt_counts[idx_gt[0]] if len(idx_gt) > 0 else 0
            
            class_names.append(CLASS_MAP.get(cls, f"Class {cls}"))
            colors.append(color_map.get(cls, [128, 128, 128]))
        
        # Convert RGB colors to matplotlib format
        colors_matplotlib = [[c/255 for c in color] for color in colors]
        
        # Plot pie charts side by side
        axs[1, 2].pie(pred_dist, labels=class_names, colors=colors_matplotlib, autopct='%1.1f%%')
        axs[1, 2].set_title('Predicted Class Distribution')
    
    # Calculate and add accuracy information if ground truth is available
    if gt_mask is not None:
        accuracy = np.mean(pred_mask == gt_mask) * 100
        fig.suptitle(f'Prediction Visualization (Accuracy: {accuracy:.2f}%)', fontsize=16)
        
        # Add per-class accuracy
        for cls in all_classes:
            cls_name = CLASS_MAP.get(cls, f"Class {cls}")
            cls_mask_gt = gt_mask == cls
            if np.sum(cls_mask_gt) > 0:
                cls_accuracy = np.mean(pred_mask[cls_mask_gt] == cls) * 100
                fig.text(0.02, 0.85 - cls*0.03, 
                        f"{cls_name} Accuracy: {cls_accuracy:.2f}%", 
                        fontsize=10)
    else:
        fig.suptitle('Prediction Visualization', fontsize=16)
    
    # Add legend
    patches = []
    for cls, color in color_map.items():
        if cls in CLASS_MAP:
            color_norm = [c/255 for c in color]
            patches.append(mpatches.Patch(color=color_norm, label=CLASS_MAP[cls]))
    
    # Position the legend under the subplots
    fig.legend(handles=patches, loc='lower center', ncol=len(color_map), fontsize=12, bbox_to_anchor=(0.5, 0.05))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save figure if path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved detailed visualization to {output_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Predict segmentation masks using SegFormer model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--model_name', type=str, default='nvidia/mit-b0', help='Name of the SegFormer model')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of segmentation classes')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for predictions')
    parser.add_argument('--img_size', type=int, default=512, help='Size to resize images to')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--no_indices', action='store_true', help='Do not add spectral indices')
    parser.add_argument('--is_sentinel', action='store_true', help='Treat input as Sentinel-2 data')
    parser.add_argument('--mask_dir', type=str, default=None, help='Optional: Directory with ground truth masks for comparison')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    
    # Determine expected input channels from the model state dict
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        
        # Check the size of the first layer's weight parameter
        for key in state_dict.keys():
            if 'patch_embeddings.0.proj.weight' in key:
                expected_channels = state_dict[key].shape[1]
                print(f"Model loaded from {args.model_path} expects {expected_channels} input channels")
                break
    except Exception as e:
        print(f"Couldn't determine expected channels from model: {e}")
        expected_channels = 5  # Default to 5 channels (RGB + 2 indices)

    # Create a transform without augmentations
    transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        ToTensorV2(),
    ], is_check_shapes=False)  # Disable shape consistency check

    # Initialize model architecture
    print(f"Initializing model architecture with {args.num_classes} classes...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True,
        id2label={str(i): f"class_{i}" for i in range(args.num_classes)},
        label2id={f"class_{i}": str(i) for i in range(args.num_classes)}
    )
    
    # Adjust model's first layer to match expected input channels
    current_channels = model.segformer.encoder.patch_embeddings[0].proj.weight.shape[1]
    
    if current_channels != expected_channels:
        print(f"Adjusting model architecture to match expected input channels ({current_channels} -> {expected_channels})")
        
        # Create a new convolutional layer with the correct input channels
        original_conv = model.segformer.encoder.patch_embeddings[0].proj
        new_conv = torch.nn.Conv2d(
            in_channels=expected_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # Initialize new weights, copying over weights for RGB channels and initializing new channels
        if current_channels >= 3 and expected_channels >= 3:
            # Copy RGB weights
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight[:, :3, :, :]
                
                # Initialize remaining channels as average of RGB channels or with new values
                if expected_channels > 3:
                    new_conv.weight[:, 3:, :, :] = torch.mean(original_conv.weight[:, :3, :, :], dim=1, keepdim=True).repeat(1, expected_channels-3, 1, 1)
                    
        # Replace the convolutional layer
        model.segformer.encoder.patch_embeddings[0].proj = new_conv
    
    # Load weights from pre-trained model
    print(f"Loading model weights from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=device)
    
    # If the model's first layer has been modified, remove its state_dict entry
    if current_channels != expected_channels:
        # Get the key prefix
        keys_to_remove = []
        for key in state_dict.keys():
            if 'patch_embeddings.0.proj.weight' in key or 'patch_embeddings.0.proj.bias' in key:
                keys_to_remove.append(key)
                
        # Remove the keys
        for key in keys_to_remove:
            print(f"Removing key from state dict: {key}")
            del state_dict[key]
    
    # Load state dict, allowing for missing keys (due to the removed first layer)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Predict masks
    if os.path.isdir(args.input):
        # If input is a directory, process all images
        predict_directory(
            model, 
            args.input, 
            args.output_dir, 
            transform, 
            device, 
            args.num_classes,
            args.img_size,
            not args.no_indices,
            args.is_sentinel
        )
    else:
        # If input is a single image, process it
        print(f"Processing single image: {args.input}")
        
        # Create dataset with the specified parameters
        pred_mask, img_filename = predict_mask(
            model, 
            args.input, 
            transform, 
            device, 
            args.num_classes,
            args.img_size,
            not args.no_indices,
            args.is_sentinel
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save masks
        image_name = os.path.splitext(os.path.basename(args.input))[0]
        mask_path = os.path.join(args.output_dir, f"{image_name}_mask.png")
        colored_mask_path = os.path.join(args.output_dir, f"{image_name}_mask_colored.png")
        viz_path = os.path.join(args.output_dir, f"{image_name}_viz.png")
        
        # Save raw class mask
        cv2.imwrite(mask_path, pred_mask)
        print(f"Saved mask to {mask_path}")
        
        # Create and save colored mask
        colored_mask = create_colored_mask(pred_mask)
        cv2.imwrite(colored_mask_path, colored_mask)
        print(f"Saved colored mask to {colored_mask_path}")
        
        # Create and save visualization
        original_image = cv2.imread(args.input)
        original_image = cv2.resize(original_image, (pred_mask.shape[1], pred_mask.shape[0]))
        
        # Try to find ground truth mask for comparison
        gt_mask, gt_mask_path = check_and_load_ground_truth(
            os.path.basename(args.input), 
            args.mask_dir or os.path.join(os.path.dirname(os.path.dirname(args.input)), 'masks')
        )
        
        # Create detailed visualization with comparison to ground truth if available
        detailed_viz_path = os.path.join(args.output_dir, f"{image_name}_detailed.png")
        create_comparison_visualization(
            original_image, 
            pred_mask, 
            gt_mask, 
            detailed_viz_path
        )
        
        # Create a simple overlay for backward compatibility
        alpha = 0.5
        viz = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
        cv2.imwrite(viz_path, viz)
        print(f"Saved visualization to {viz_path}")
        
        print(f"All predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main()