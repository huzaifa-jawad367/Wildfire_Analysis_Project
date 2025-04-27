import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

def balance_model_weights(model_path, output_path, class_names=None):
    """
    Adjust a model's weights to better balance class predictions
    """
    if class_names is None:
        class_names = ["Background", "Water", "Terrain", "Vegetation", "Forest", "Cloud"]
    
    print(f"Loading model from {model_path}")
    
    # Load state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if classifier weights exist
    if 'decode_head.classifier.weight' in state_dict and 'decode_head.classifier.bias' in state_dict:
        classifier_weights = state_dict['decode_head.classifier.weight']
        classifier_bias = state_dict['decode_head.classifier.bias']
        
        print(f"\nOriginal classifier layer shape: {classifier_weights.shape}")
        print(f"Number of classes: {classifier_weights.shape[0]}")
        
        # Bias before adjustment
        print("\nBias values before adjustment:")
        for i in range(classifier_bias.shape[0]):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"  Class {i} ({class_name}): {classifier_bias[i].item():.6f}")
        
        # Calculate statistics to identify dominant classes
        orig_relative_strength = classifier_bias.abs() / classifier_bias.abs().mean()
        print("\nRelative bias strength before adjustment:")
        for i in range(orig_relative_strength.shape[0]):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"  Class {i} ({class_name}): {orig_relative_strength[i].item():.4f}x average")
        
        # Adjust the classifier weights and biases
        
        # 1. Normalize bias values to balance classes
        # - Reduce bias of dominant classes (usually background, water, terrain)
        # - Increase bias of underrepresented classes (vegetation, forest, cloud)
        
        dominant_classes = [0, 1, 2]  # Background, Water, Terrain are often dominant
        underrepresented_classes = [3, 4, 5]  # Vegetation, Forest, Cloud are often underrepresented
        
        # Adjust biases
        new_classifier_bias = classifier_bias.clone()
        
        # Reduce bias of dominant classes
        for i in dominant_classes:
            if i < new_classifier_bias.shape[0]:
                new_classifier_bias[i] *= 0.7  # Reduce bias by 30%
        
        # Increase bias of underrepresented classes
        for i in underrepresented_classes:
            if i < new_classifier_bias.shape[0]:
                new_classifier_bias[i] += 0.5  # Boost bias
        
        # 2. Adjust weights for better feature extraction
        new_classifier_weights = classifier_weights.clone()
        
        # Strengthen feature detection for underrepresented classes
        for i in underrepresented_classes:
            if i < new_classifier_weights.shape[0]:
                scale_factor = 1.2  # Increase weight magnitude by 20%
                new_classifier_weights[i] = new_classifier_weights[i] * scale_factor
        
        # Apply the adjusted weights to the state dict
        state_dict['decode_head.classifier.bias'] = new_classifier_bias
        state_dict['decode_head.classifier.weight'] = new_classifier_weights
        
        # Show the adjusted bias values
        print("\nBias values after adjustment:")
        for i in range(new_classifier_bias.shape[0]):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"  Class {i} ({class_name}): {new_classifier_bias[i].item():.6f}")
        
        # Calculate new relative strength
        new_relative_strength = new_classifier_bias.abs() / new_classifier_bias.abs().mean()
        print("\nRelative bias strength after adjustment:")
        for i in range(new_relative_strength.shape[0]):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(f"  Class {i} ({class_name}): {new_relative_strength[i].item():.4f}x average")
        
        # Save the adjusted model
        print(f"\nSaving balanced model to {output_path}")
        torch.save(state_dict, output_path)
        print("Model saved successfully!")
        
        return True
    else:
        print("Could not find classifier weights in the model")
        return False

def main():
    parser = argparse.ArgumentParser(description="Balance a trained segmentation model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--output_path", type=str, default="balanced_model.pth", help="Path to save the balanced model")
    args = parser.parse_args()
    
    balance_model_weights(args.model_path, args.output_path)

if __name__ == "__main__":
    main() 