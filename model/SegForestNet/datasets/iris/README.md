# Dataset Organization Guide

## Directory Structure
```
iris/
├── images/          # Contains all input images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── masks/           # Contains all segmentation masks
│   ├── image1_mask.png
│   ├── image2_mask.png
│   └── ...
├── train.txt        # List of training images (created automatically)
├── val.txt          # List of validation images (created automatically)
└── test.txt         # List of test images (created automatically)
```

## Image Requirements

### Input Images (in `images/` folder)
- Format: JPG, JPEG, or PNG
- Color: RGB
- Size: Any size (will be resized to 256x256 during training)
- Naming: Any name (e.g., `image1.jpg`, `sample_001.png`)

### Mask Images (in `masks/` folder)
- Format: PNG only
- Color: Grayscale
- Size: Same as corresponding input image
- Values: 0 to (num_classes - 1)
- Naming: Must match input image name + "_mask.png"
  - For input `image1.jpg` → mask should be `image1_mask.png`
  - For input `sample_001.png` → mask should be `sample_001_mask.png`

## Class Labels
Default configuration (6 classes):
- 0: Background
- 1: Class 1
- 2: Class 2
- 3: Class 3
- 4: Class 4
- 5: Class 5

To modify the number of classes:
1. Edit `configs/config.yaml`
2. Change `num_classes` under `SegForestNet_params`
3. Update mask images to use correct class values

## Data Splits
- Training set: 80% of images
- Validation set: 10% of images
- Test set: 10% of images

Split files (train.txt, val.txt, test.txt) will be created automatically when you first run the training script. 