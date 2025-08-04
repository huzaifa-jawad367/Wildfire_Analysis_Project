import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from transformers import SegformerForSemanticSegmentation

def analyze_model_weights(model_path):
    """Analyze a model's weights to understand class bias"""
    print(f"Loading model from {model_path}")
    
    # Load state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    
    # Check input channels
    if 'segformer.encoder.patch_embeddings.0.proj.weight' in state_dict:
        in_channels = state_dict['segformer.encoder.patch_embeddings.0.proj.weight'].shape[1]
        print(f"Model expects {in_channels} input channels")
    
    # Analyze final classification layer
    if 'decode_head.classifier.weight' in state_dict:
        classifier_weights = state_dict['decode_head.classifier.weight']
        classifier_bias = state_dict['decode_head.classifier.bias']
        
        print(f"\nClassifier layer shape: {classifier_weights.shape}")
        print(f"Number of classes: {classifier_weights.shape[0]}")
        
        # Analyze weight statistics per class
        print("\nWeight statistics per class:")
        for i in range(classifier_weights.shape[0]):
            class_weights = classifier_weights[i].flatten()
            print(f"  Class {i}: mean={class_weights.mean().item():.6f}, std={class_weights.std().item():.6f}, "
                  f"min={class_weights.min().item():.6f}, max={class_weights.max().item():.6f}")
        
        # Analyze bias values
        print("\nBias values per class:")
        for i in range(classifier_bias.shape[0]):
            print(f"  Class {i}: {classifier_bias[i].item():.6f}")
        
        # Check if there's a significant bias towards certain classes
        print("\nRelative bias strength (higher values may dominate predictions):")
        relative_strength = classifier_bias.abs() / classifier_bias.abs().mean()
        for i in range(relative_strength.shape[0]):
            print(f"  Class {i}: {relative_strength[i].item():.4f}x average")
            
        # Visualize weight distributions
        plt.figure(figsize=(10, 6))
        for i in range(classifier_weights.shape[0]):
            weights = classifier_weights[i].flatten().cpu().numpy()
            plt.hist(weights, alpha=0.5, bins=50, label=f"Class {i}")
        
        plt.title("Classifier Weight Distributions by Class")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("model_weight_analysis.png")
        print(f"\nSaved weight distribution visualization to model_weight_analysis.png")
    else:
        print("Could not find classifier weights in the model")
    
    # Check if any layers have abnormal weight patterns
    outlier_layers = []
    for name, param in state_dict.items():
        if param.dim() > 1:  # Only analyze multi-dimensional tensors
            param_flat = param.flatten()
            if param_flat.shape[0] > 10:  # Skip tiny tensors
                mean = param_flat.mean().item()
                std = param_flat.std().item()
                if std < 1e-6 or abs(mean) > 1.0:
                    outlier_layers.append((name, mean, std))
    
    if outlier_layers:
        print("\nPotential problematic layers (very low variance or high bias):")
        for name, mean, std in outlier_layers:
            print(f"  {name}: mean={mean:.6f}, std={std:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a trained segmentation model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights file")
    args = parser.parse_args()
    
    analyze_model_weights(args.model_path)

if __name__ == "__main__":
    main() 