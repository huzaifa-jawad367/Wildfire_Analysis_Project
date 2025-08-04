import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
import logging

def plot_training_progress(train_losses, val_losses, save_path):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def visualize_prediction(model, image, mask, output_path):
    """Visualize model prediction alongside ground truth"""
    device = next(model.parameters()).device
    
    # Ensure model is in eval mode
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
        else:
            image_tensor = image.float()
            
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Convert mask to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if image_tensor.shape[1] == 3:
        img_display = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(img_display)
    else:
        axes[0].imshow(image_tensor[0, 0].cpu().numpy(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask, cmap='tab20')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred, cmap='tab20')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_training_summary(output_dir):
    """Create a visual summary of the training run"""
    try:
        # Load training history
        train_losses = np.load(os.path.join(output_dir, 'train_losses.npy'))
        val_losses = np.load(os.path.join(output_dir, 'val_losses.npy'))
        
        # Plot losses
        plot_training_progress(
            train_losses, 
            val_losses, 
            os.path.join(output_dir, 'training_progress.png')
        )
        
        # Create summary text
        summary = [
            "Training Summary",
            "================",
            f"Total epochs: {len(train_losses)}",
            f"Best validation loss: {min(val_losses):.4f}",
            f"Final training loss: {train_losses[-1]:.4f}",
            f"Final validation loss: {val_losses[-1]:.4f}",
            f"Loss reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%"
        ]
        
        # Save summary
        with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))
            
        logging.info("Training summary created successfully")
        
    except Exception as e:
        logging.error(f"Error creating training summary: {str(e)}")

def visualize_dataset_sample(dataset, output_dir, num_samples=5):
    """Visualize random samples from the dataset"""
    try:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for idx in indices:
            image, mask = dataset[idx]
            
            # Create visualization
            plt.figure(figsize=(10, 5))
            
            # Display image
            plt.subplot(1, 2, 1)
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:
                    plt.imshow(image.numpy().transpose(1, 2, 0))
                else:
                    plt.imshow(image.numpy()[0], cmap='gray')
            else:
                plt.imshow(image)
            plt.title('Image')
            plt.axis('off')
            
            # Display mask
            plt.subplot(1, 2, 2)
            if isinstance(mask, torch.Tensor):
                plt.imshow(mask.numpy(), cmap='tab20')
            else:
                plt.imshow(mask, cmap='tab20')
            plt.title('Mask')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'dataset_sample_{idx}.png'))
            plt.close()
            
        logging.info(f"Created {num_samples} dataset visualizations")
        
    except Exception as e:
        logging.error(f"Error visualizing dataset: {str(e)}")

def plot_confusion_matrix(true_labels, pred_labels, classes, output_path):
    """Plot confusion matrix for segmentation results"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(true_labels.flatten(), pred_labels.flatten())
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 