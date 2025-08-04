import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random
from PIL import Image

class SegmentationTransform:
    def __init__(self, config):
        self.config = config
        
    def __call__(self, image):
        # Random rotation
        if self.config['enable'] and random.random() > 0.5:
            angle = random.uniform(-self.config['rotation_range'], 
                                 self.config['rotation_range'])
            image = F.rotate(image, angle)
        
        # Random horizontal flip
        if self.config['enable'] and random.random() > 0.5:
            image = F.hflip(image)
            
        # Random vertical flip
        if self.config['enable'] and random.random() > 0.5:
            image = F.vflip(image)
            
        # Random zoom
        if self.config['enable'] and random.random() > 0.5:
            scale = random.uniform(1 - self.config['zoom_range'],
                                 1 + self.config['zoom_range'])
            size = image.size if isinstance(image, Image.Image) else image.shape[-2:]
            new_size = [int(scale * s) for s in size]
            image = F.resize(image, new_size)
            
        return image

def get_transforms(config):
    """
    Returns the transformation pipeline based on config
    
    Args:
        config (dict): Configuration for data augmentation
    
    Returns:
        transform (callable): Transformation pipeline
    """
    if not config['enable']:
        return None  # No additional transforms
    
    return SegmentationTransform(config) 