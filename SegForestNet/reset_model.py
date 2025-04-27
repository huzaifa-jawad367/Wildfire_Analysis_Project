import os
import torch
import argparse
import numpy as np
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

def reset_model_weights(model_path, output_path, num_classes=6, num_channels=5):
    """Reset model classifier weights to avoid class collapse"""
    print(f"Loading model from {model_path}")
    
    # Try to load the model or start fresh
    try:
        # Load state dict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=device)
        
        # Check input channels
        if 'segformer.encoder.patch_embeddings.0.proj.weight' in state_dict:
            in_channels = state_dict['segformer.encoder.patch_embeddings.0.proj.weight'].shape[1]
            print(f"Model expects {in_channels} input channels")
        
        # Create a fresh model
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Modify first layer if needed
        if num_channels != 3:
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
        
        # Reset the classifier weights with more balanced initialization
        with torch.no_grad():
            # Get the shape of the classifier weights
            if 'decode_head.classifier.weight' in state_dict:
                class_shape = state_dict['decode_head.classifier.weight'].shape
                bias_shape = state_dict['decode_head.classifier.bias'].shape
                
                # Create new weights with more balanced initialization
                # Initialize weights from a normal distribution with class-specific parameters
                new_weights = torch.zeros(class_shape, device=device)
                new_bias = torch.zeros(bias_shape, device=device)
                
                # Set bias values to favor different classes equally
                # Using small positive values for non-terrain classes
                for i in range(num_classes):
                    if i == 2:  # Terrain class
                        new_bias[i] = 0.0  # Neutral bias for terrain
                    else:
                        new_bias[i] = 0.2  # Positive bias for other classes
                
                # Initialize weights for each class with different scales
                for i in range(num_classes):
                    # Scale based on class importance
                    scale = 0.02
                    if i in [1, 3, 4, 5]:  # Water, Veg, Forest, Cloud
                        scale = 0.03  # Slightly higher scale for underrepresented classes
                    
                    # Initialize with normal distribution
                    new_weights[i] = torch.normal(
                        mean=0.0, 
                        std=scale, 
                        size=new_weights[i].shape,
                        device=device
                    )
                
                # Update the state dict
                state_dict['decode_head.classifier.weight'] = new_weights
                state_dict['decode_head.classifier.bias'] = new_bias
        
        # Save the updated model
        print(f"Saving reinitialized model to {output_path}")
        torch.save(state_dict, output_path)
        print("Model saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error resetting model weights: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Reset SegFormer model classifier weights")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model weights file")
    parser.add_argument("--output_path", type=str, default="reset_model.pth", 
                        help="Path to save the reset model")
    parser.add_argument("--num_classes", type=int, default=6,
                       help="Number of segmentation classes")
    parser.add_argument("--num_channels", type=int, default=5,
                       help="Number of input channels (3 for RGB, 5 for RGB+indices)")
    args = parser.parse_args()
    
    reset_model_weights(args.model_path, args.output_path, args.num_classes, args.num_channels)

if __name__ == "__main__":
    main() 