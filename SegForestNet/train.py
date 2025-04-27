import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from transformers import SegformerForSemanticSegmentation
from dataset import get_train_val_dataloaders
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from skimage.segmentation import slic
from sklearn.metrics import confusion_matrix
from dataset import MultiClassLandCoverDataset

def calculate_metrics(pred_masks, true_masks, num_classes):
    """Calculate various metrics for evaluation."""
    metrics = {}
    
    # Convert tensors to numpy arrays for calculations
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(true_masks, torch.Tensor):
        true_masks = true_masks.cpu().numpy()
    
    # For multi-class, calculate metrics per class
    class_iou = []
    class_dice = []
    
    # Reshape to (-1) to create 1D arrays
    pred_flat = pred_masks.reshape(-1)
    true_flat = true_masks.reshape(-1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_flat, pred_flat, labels=range(num_classes))
    
    # Calculate metrics for each class
    for i in range(num_classes):
        # True positives, false positives, false negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        # Calculate IoU and Dice
        iou = tp / (tp + fp + fn + 1e-10)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
        
        class_iou.append(iou)
        class_dice.append(dice)
    
    # Overall metrics
    metrics['mean_iou'] = np.mean(class_iou)
    metrics['mean_dice'] = np.mean(class_dice)
    metrics['class_iou'] = class_iou
    metrics['class_dice'] = class_dice
    
    # Calculate accuracy
    accuracy = np.sum(pred_flat == true_flat) / len(true_flat)
    metrics['accuracy'] = accuracy
    
    return metrics

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
    if len(pred_np.shape) > 2:  # Multi-class predictions
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

def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_dataset = MultiClassLandCoverDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        img_size=args.img_size,
        inference=False,
        num_classes=args.num_classes,
        use_indices=args.use_indices,
        is_sentinel=args.is_sentinel
    )
    
    # Analyze mask class distribution and get automatic weights
    auto_class_weights = train_dataset.analyze_masks(args.mask_dir)
    
    # Create train/val dataloaders
    train_loader, val_loader = get_train_val_dataloaders(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        img_size=args.img_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        use_indices=args.use_indices,
        is_sentinel=args.is_sentinel
    )
    
    # Initialize SegFormer model
    print("Initializing SegFormer model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,  # Multi-class segmentation
        ignore_mismatched_sizes=True  # Needed for the different output classes
    )
    
    # Handle the case where the input has more than 3 channels (like Sentinel-2 with indices)
    # Get a sample from the dataset to check input channels
    sample_batch = next(iter(train_loader))
    num_channels = sample_batch["image"].shape[1]
    
    if num_channels > 3:
        print(f"Input has {num_channels} channels. Modifying model to accept multi-spectral input.")
        
        # Get the original segformer weights
        original_conv = model.segformer.encoder.patch_embeddings[0].proj
        
        # Create new conv layer with the desired number of input channels but same output channels
        new_conv = nn.Conv2d(
            num_channels, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # Initialize the new_conv with the weights from the trained model
        # For the first 3 channels, copy the weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight
            
            # For additional channels, initialize with the mean of RGB channels
            if num_channels > 3:
                channel_mean = original_conv.weight.mean(dim=1, keepdim=True)
                for c in range(3, num_channels):
                    new_conv.weight[:, c:c+1, :, :] = channel_mean
            
            # Copy bias if it exists
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias
        
        # Replace the conv in the model
        model.segformer.encoder.patch_embeddings[0].proj = new_conv
    
    model.to(device)
    
    # Define loss function and optimizer
    if args.num_classes > 1:
        # For multi-class segmentation with class weighting
        # Use automatically calculated weights if available, otherwise use default weights
        if auto_class_weights:
            # Convert dict to tensor
            auto_weights = torch.ones(args.num_classes, device=device)
            for i, w in auto_class_weights.items():
                auto_weights[i] = w
            print(f"Using automatically calculated class weights: {auto_weights}")
            criterion = nn.CrossEntropyLoss(weight=auto_weights)
        else:
            # Define weights to give more importance to less common classes
            # Increase weights for underrepresented classes significantly
            class_weights = torch.tensor([1.0, 4.0, 0.7, 3.0, 3.0, 4.0], device=device)  # Background, Water, Terrain, Vegetation, Forest, Cloud
            print(f"Using default class weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # For binary segmentation
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler - use cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=1,  # Don't increase restart interval
        eta_min=args.learning_rate * 0.01  # Minimum learning rate
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0
    best_mean_iou = 0.0
    class_balance_history = []
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_metrics = []
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Forward pass
            outputs = model(pixel_values=images).logits
            
            # SegFormer outputs smaller resolution, resize masks to match
            if outputs.shape[2:] != masks.shape[2:]:
                if args.num_classes > 1:
                    # For multi-class, first ensure all mask values are valid
                    masks = torch.clamp(masks, 0, args.num_classes - 1)
                    # Convert to one-hot, then interpolate, then back to class indices
                    masks = F.interpolate(
                        F.one_hot(masks.long(), num_classes=args.num_classes).permute(0, 3, 1, 2).float(),
                        size=outputs.shape[2:],
                        mode='nearest'
                    )
                    masks = masks.argmax(dim=1)  # Convert back to class indices
                else:
                    # For binary, just resize the mask
                    masks = F.interpolate(masks.float(), size=outputs.shape[2:], mode='nearest')
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": train_loss / train_steps})
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                
                # Forward pass
                outputs = model(pixel_values=images).logits
                
                # SegFormer outputs smaller resolution, resize masks to match
                if outputs.shape[2:] != masks.shape[2:]:
                    if args.num_classes > 1:
                        # For multi-class, first ensure all mask values are valid
                        masks = torch.clamp(masks, 0, args.num_classes - 1)
                        # Convert to one-hot, then interpolate, then back to class indices
                        masks = F.interpolate(
                            F.one_hot(masks.long(), num_classes=args.num_classes).permute(0, 3, 1, 2).float(),
                            size=outputs.shape[2:],
                            mode='nearest'
                        )
                        masks = masks.argmax(dim=1)  # Convert back to class indices
                    else:
                        # For binary, just resize the mask
                        masks = F.interpolate(masks.float(), size=outputs.shape[2:], mode='nearest')
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Get predictions
                if args.num_classes > 1:
                    # For multi-class, get class with highest probability
                    preds = outputs.argmax(dim=1)
                else:
                    # For binary, threshold sigmoid output
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                
                # Apply post-processing if enabled
                if args.use_postprocessing:
                    smoothed_preds = []
                    for i in range(preds.size(0)):
                        smoothed_pred = post_process_prediction(
                            preds[i], images[i], n_segments=args.postprocessing_segments
                        )
                        smoothed_preds.append(smoothed_pred)
                    preds = torch.stack(smoothed_preds)
                
                # Calculate metrics
                batch_metrics = calculate_metrics(preds, masks, args.num_classes)
                all_metrics.append(batch_metrics)
                
                # Monitor class distribution to detect class collapse
                class_counts = []
                for c in range(args.num_classes):
                    count = (preds == c).sum().item()
                    class_counts.append(count)
                total_pixels = preds.numel()
                
                # Save class distribution for tracking
                class_distribution = [(count / total_pixels) * 100 for count in class_counts]
                class_balance_history.append(class_distribution)
                
                # Check for class collapse (one class dominates predictions)
                if epoch % 5 == 0:  # Check every 5 epochs
                    print("\nClass distribution in validation predictions:")
                    for c in range(args.num_classes):
                        percentage = (class_counts[c] / total_pixels) * 100
                        print(f"  Class {c}: {class_counts[c]} pixels ({percentage:.2f}%)")
                    
                    # Alert if class collapse is detected
                    max_class_percent = max([(c / total_pixels) * 100 for c in class_counts])
                    if max_class_percent > 90:
                        print(f"WARNING: Potential class collapse detected. One class represents {max_class_percent:.2f}% of predictions!")
                        print("Consider increasing class weights for underrepresented classes or using data augmentation.")
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # Calculate average metrics across all validation batches
        epoch_metrics = {}
        for metric in all_metrics[0].keys():
            if isinstance(all_metrics[0][metric], list):
                # Handle lists (like per-class metrics)
                epoch_metrics[metric] = [
                    np.mean([m[metric][i] for m in all_metrics])
                    for i in range(len(all_metrics[0][metric]))
                ]
            else:
                # Handle scalar metrics
                epoch_metrics[metric] = np.mean([m[metric] for m in all_metrics])
        
        val_metrics.append(epoch_metrics)
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val IoU: {epoch_metrics['mean_iou']:.4f}, "
              f"Val Dice: {epoch_metrics['mean_dice']:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving best model with validation loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), args.output_model_path)
            
        # Save visualization of predictions at various epochs
        if (epoch + 1) % args.viz_frequency == 0 or epoch == 0 or epoch == args.num_epochs - 1:
            save_validation_samples(model, val_loader, device, epoch, args.output_dir, args.num_classes)
        
        # Early stopping logic
        if epoch_metrics['mean_iou'] > best_mean_iou:
            best_mean_iou = epoch_metrics['mean_iou']
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break
    
    # Plot and save training curves
    plt.figure(figsize=(15, 10))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot IoU curves
    plt.subplot(2, 2, 2)
    epochs = range(len(val_metrics))
    plt.plot(epochs, [m['mean_iou'] for m in val_metrics], label='Mean IoU')
    for i in range(args.num_classes):
        plt.plot(epochs, [m['class_iou'][i] for m in val_metrics], label=f'Class {i} IoU')
    plt.title('IoU Curves')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    # Plot class distribution over time
    if class_balance_history:
        plt.subplot(2, 2, 3)
        epochs = range(len(class_balance_history))
        for i in range(args.num_classes):
            class_percentages = [dist[i] for dist in class_balance_history]
            plt.plot(epochs, class_percentages, label=f'Class {i}')
        plt.title('Class Distribution Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Percentage of Pixels')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.close()
    
    # Also save metrics as CSV
    with open(os.path.join(args.output_dir, "training_metrics.csv"), "w") as f:
        # Write header
        f.write("epoch,train_loss,val_loss,mean_iou,mean_dice,accuracy")
        for i in range(args.num_classes):
            f.write(f",iou_class{i},dice_class{i}")
        f.write("\n")
        
        # Write data for each epoch
        for epoch in range(len(train_losses)):
            metrics = val_metrics[epoch]
            f.write(f"{epoch},{train_losses[epoch]},{val_losses[epoch]},{metrics['mean_iou']},{metrics['mean_dice']},{metrics['accuracy']}")
            
            # Write per-class metrics
            for i in range(args.num_classes):
                f.write(f",{metrics['class_iou'][i]},{metrics['class_dice'][i]}")
            f.write("\n")
    
    return model

def save_validation_samples(model, val_loader, device, epoch, output_dir, num_classes=6):
    """Save a few validation samples with predictions"""
    model.eval()
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Define colors for visualization
    colors = [
        [0, 0, 0],       # Background - Black
        [0, 0, 255],     # Water - Blue
        [139, 69, 19],   # Terrain - Brown
        [0, 255, 0],     # Vegetation - Green
        [0, 100, 0],     # Forest - Dark Green
        [255, 255, 255]  # Cloud - White
    ]
    
    # Ensure we have colors for all classes
    while len(colors) < num_classes:
        colors.append([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
    
    with torch.no_grad():
        # Get a batch of validation data
        batch = next(iter(val_loader))
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        filenames = batch["filename"]
        
        # Get predictions
        outputs = model(pixel_values=images).logits
        
        # For multi-class predictions
        if num_classes > 1:
            # Get class with highest probability
            preds = outputs.argmax(dim=1)
            
            # If masks and outputs are different sizes, resize outputs to match masks
            if outputs.shape[2:] != masks.shape[2:]:
                outputs_resized = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                preds_resized = outputs_resized.argmax(dim=1)
            else:
                preds_resized = preds
                
        else:
            # Binary segmentation
            # SegFormer outputs may be different size, resize to original
            if outputs.shape[2:] != masks.shape[2:]:
                outputs_resized = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            else:
                outputs_resized = outputs
                
            # Convert logits to binary predictions
            preds = torch.sigmoid(outputs) > 0.5
            preds_resized = torch.sigmoid(outputs_resized) > 0.5
        
        # Save predictions for inspection
        for i in range(min(4, len(images))):
            # Convert tensors to numpy arrays
            image = images[i].cpu().permute(1, 2, 0).numpy()
            
            # Use only RGB channels for visualization if we have more
            if image.shape[2] > 3:
                image = image[:, :, :3]
                
            # Un-normalize the image
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
            
            # Create colored masks for visualization
            if num_classes > 1:
                # Multi-class masks
                mask = masks[i].cpu().numpy()
                pred = preds_resized[i].cpu().numpy()
                
                # Create RGB colored masks
                mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
                pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
                
                # Fill in colors for each class
                for class_idx in range(num_classes):
                    mask_colored[mask == class_idx] = np.array(colors[class_idx]) / 255.0
                    pred_colored[pred == class_idx] = np.array(colors[class_idx]) / 255.0
            else:
                # Binary masks
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds_resized[i].cpu().squeeze().numpy()
                
                # Use simple green for positive class
                mask_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
                mask_colored[mask > 0.5] = np.array([0, 1, 0])  # Green for forest
                
                pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
                pred_colored[pred > 0.5] = np.array([0, 1, 0])  # Green for forest
            
            # Create plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot image
            axes[0].imshow(image)
            axes[0].set_title("Input Image")
            axes[0].axis("off")
            
            # Plot ground truth mask
            axes[1].imshow(mask_colored)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            
            # Plot predicted mask
            axes[2].imshow(pred_colored)
            axes[2].set_title("Prediction")
            axes[2].axis("off")
            
            # Add legend for multi-class
            if num_classes > 1:
                class_names = ["Background", "Water", "Terrain", "Vegetation", "Forest", "Cloud"]
                patches = [plt.Rectangle((0, 0), 1, 1, fc=np.array(color)/255) for color in colors[:len(class_names)]]
                axes[1].legend(patches, class_names, loc='lower right', fontsize='small')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"epoch{epoch+1}_{filenames[i]}.png"))
            plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train SegFormer for land cover segmentation")
    
    # Dataset parameters
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing training images")
    parser.add_argument("--mask_dir", type=str, required=True, 
                        help="Directory containing mask images")
    parser.add_argument("--img_size", type=int, default=512, 
                        help="Image size for training (default: 512)")
    parser.add_argument("--is_sentinel", action="store_true",
                        help="Treat input as Sentinel-2 multi-spectral imagery")
    parser.add_argument("--use_indices", action="store_true",
                        help="Extract and use spectral indices as additional channels")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="nvidia/mit-b0", 
                        help="SegFormer model name from HuggingFace (default: nvidia/mit-b0)")
    parser.add_argument("--num_classes", type=int, default=6,
                        help="Number of segmentation classes (default: 6)")
    parser.add_argument("--output_model_path", type=str, default="segformer_landcover.pth", 
                        help="Path to save the trained model")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Directory to save outputs")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for training (default: 2)")
    parser.add_argument("--num_epochs", type=int, default=30, 
                        help="Number of epochs for training (default: 30)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="Weight decay (default: 1e-5)")
    parser.add_argument("--val_split", type=float, default=0.1, 
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of workers for data loading (default: 0)")
    
    # Visualization parameters
    parser.add_argument("--viz_frequency", type=int, default=5,
                        help="Save visualizations every N epochs (default: 5)")
    
    # Post-processing parameters
    parser.add_argument("--use_postprocessing", action="store_true",
                        help="Apply SLIC-based post-processing to predictions")
    parser.add_argument("--postprocessing_segments", type=int, default=100,
                        help="Number of segments for SLIC post-processing (default: 100)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    model = train_model(args) 