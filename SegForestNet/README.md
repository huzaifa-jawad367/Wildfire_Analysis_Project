# SegFormer Land Cover Segmentation

This project uses the SegFormer architecture (specifically SegFormer-B0) to segment land cover types from aerial/satellite imagery, including water, terrain, vegetation, and forest. It is optimized for CPU-only execution on Intel Iris Graphics.

## Features

- Multi-class land cover segmentation (5 classes: background, water, terrain, vegetation, forest)
- Spectral indices extraction to enhance vegetation detection capabilities 
- SLIC-based post-processing for smoother, more coherent segmentation
- CPU-optimized implementation using Hugging Face Transformers
- Data augmentation techniques for improved model generalization
- Comprehensive metrics tracking (IoU, Dice, accuracy)
- Visualization tools for model predictions

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:

```
data/
├── images/          # RGB aerial/satellite images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/           # Multi-class masks or binary masks
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

For multi-class masks, the following color scheme is interpreted:
- Black (0,0,0): Background
- Blue (dominant): Water
- Brown/Gray: Terrain
- Bright Green: Vegetation
- Dark Green: Forest

Binary masks are automatically converted to multi-class with forest (white) as class 4.

## Training

Train the model with:

```bash
python train.py --image_dir data/images --mask_dir data/masks --output_model_path segformer_landcover.pth
```

### Optional Arguments:

- `--img_size`: Image size for training (default: 512)
- `--model_name`: SegFormer model name from HuggingFace (default: "nvidia/mit-b0")
- `--num_classes`: Number of segmentation classes (default: 5)
- `--batch_size`: Batch size for training (default: 2)
- `--num_epochs`: Number of epochs for training (default: 30)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--val_split`: Validation split ratio (default: 0.1)
- `--output_dir`: Directory to save outputs (default: "outputs")
- `--use_postprocessing`: Apply SLIC-based post-processing to predictions
- `--postprocessing_segments`: Number of segments for SLIC (default: 100)
- `--viz_frequency`: Save visualizations every N epochs (default: 5)

## Inference

Predict masks for new images:

### Single Image:

```bash
python predict.py --model_path segformer_landcover.pth --input path/to/image.jpg
```

### Directory of Images:

```bash
python predict.py --model_path segformer_landcover.pth --input path/to/images/
```

### Optional Arguments:

- `--output_dir`: Directory to save predictions (default: "predictions")
- `--img_size`: Image size for prediction (default: 512)
- `--model_name`: SegFormer model name from HuggingFace (default: "nvidia/mit-b0")
- `--num_classes`: Number of segmentation classes (default: 5)
- `--use_postprocessing`: Apply SLIC-based post-processing to predictions
- `--postprocessing_segments`: Number of segments for SLIC (default: 100)

## Output Format

The prediction script generates:

1. Class index mask: Integer values represent different land cover classes
2. Colored mask: RGB visualization of different classes
3. Visualization: Side-by-side comparison of input image, predicted mask, and overlay

## CPU Optimization Notes

This implementation is optimized for CPU execution:
- Uses SegFormer-B0, the smallest and fastest SegFormer variant
- Configurable image size allows downscaling for faster processing
- Small batch size suitable for limited memory environments
- Efficient spectral indices extraction without requiring additional bands
- Post-processing optimized for CPU execution

## Spectral Indices

The model extracts the following spectral indices from RGB images:
- Pseudo-NDVI (using Red and Green bands as proxy)
- Enhanced Vegetation Index (simplified for RGB)
- Visible Atmospherically Resistant Index (VARI)

These indices enhance the model's ability to distinguish between vegetation types.

## Extending the Model

The implementation can be extended to:
- Support additional classes by modifying the class map in `dataset.py`
- Include true NDVI by adding support for NIR band
- Add more sophisticated post-processing techniques
- Implement ensemble methods for improved accuracy

## Credits

This project uses:
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [SegFormer](https://huggingface.co/nvidia/mit-b0) from NVIDIA
- [Albumentations](https://albumentations.ai/) for data augmentation
- [scikit-image](https://scikit-image.org/) for SLIC segmentation 