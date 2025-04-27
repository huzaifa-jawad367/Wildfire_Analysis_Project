import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F

class IRISDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Set up image and mask directories
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # Load split file
        split_file = os.path.join(root_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.images = [line.strip() for line in f.readlines()]
        
        # Define transforms
        self.input_size = (256, 256)
        self.resize_transform = T.Resize(self.input_size, interpolation=T.InterpolationMode.BILINEAR)
        self.mask_resize_transform = T.Resize(self.input_size, interpolation=T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('sat', 'mask')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Open as PIL images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply resize transform (keeping as PIL for additional transforms)
        image = self.resize_transform(image)
        mask = self.mask_resize_transform(mask)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert to tensor after all PIL-based transforms
        image = self.to_tensor(image)
        
        # For mask, convert it to a numpy array first to handle class labels properly
        mask_np = np.array(mask)
        # Normalize mask to be in the range [0, num_classes-1]
        # This assumes your mask values are already in the range [0, 255]
        # and you have 6 classes (as per config)
        mask_np = (mask_np / 255.0 * 5).astype(np.int64)  # Scale to 0-5 for 6 classes
        mask_tensor = torch.from_numpy(mask_np)
        
        # Always resize to the standard input size to ensure consistent batch dimensions
        if image.shape[1:] != self.input_size:
            image = F.interpolate(image.unsqueeze(0), size=self.input_size, 
                                mode='bilinear', align_corners=True).squeeze(0)
        
        if mask_tensor.shape != self.input_size:
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(), 
                                      size=self.input_size, mode='nearest').squeeze(0).squeeze(0).long()
        
        return image, mask_tensor

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with images of different sizes.
    Resizes all images and masks to the size of the first image in the batch.
    """
    images = []
    masks = []
    
    for image, mask in batch:
        # Ensure all have the same size as the first one
        images.append(image)
        masks.append(mask)
    
    # Stack images and masks
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return images, masks 