# Brain Tumor Classification with PyTorch

This repository contains a PyTorch-based pipeline for **brain tumor classification** from MRI images.  
It supports multiple model architectures, flexible data augmentation pipelines, metric tracking, and transfer learning with popular pretrained networks.

---

## Project Structure

```
├── models.py           # CNN architectures and transfer learning models
├── dataloader.py       # Dataset class, data transforms, and dataloader utilities
├── metrics.py          # Accuracy and IoU/Confusion Matrix metrics
├── train.py            # Training loop with logging, validation, and model saving
├── split_dataset.py    # Utility to split raw data into train/val/test
└── README.md
```

---

## Features

- **Custom CNN Architectures**:
  - `SimpleCNN`: Small, lightweight CNN for grayscale inputs.
  - `ComplexCNN`: Residual-style CNN with `GroupNorm` and flexible block stacking.

- **Transfer Learning Models**:
  - `transferResNet`: ResNet18-based classifier with frozen lower layers.
  - `transferMobileNet`: MobileNetV2-based classifier with partial fine-tuning.
  - `transferResNetLarge`: ResNet50-based classifier for larger capacity.

- **Data Augmentation**:
  - Grayscale preprocessing for scratch training.
  - ImageNet preprocessing for pretrained models.
  - Advanced augmentation for training robustness.

- **Metrics**:
  - `AccuracyMetric` for classification accuracy.
  - `ConfusionMatrix` for IoU and per-class accuracy.

- **Training Utilities**:
  - Automatic GPU/MPS support.
  - Cosine annealing learning rate scheduler (optional).
  - TensorBoard logging of images, losses, and metrics.
  - Checkpoint saving after each epoch and final model export.

---

## Dataset

The project assumes a dataset structured as:

```
raw_data/
    glioma/
        image1.jpg
        image2.jpg
        ...
    meningioma/
    notumor/
    pituitary/
```

You can split the dataset into `train`, `val`, and `test` sets using:

```bash
python split_dataset.py
```

By default, it uses a 70% / 15% / 15% split.

---

## Getting Started

### 1️ Install Dependencies
```bash
pip install torch torchvision pillow numpy tensorboard
```

### 2️ Prepare Dataset
- Place raw images in the `raw_data/` directory as described above.
- Run:
```bash
python split_dataset.py
```

### 3️ Train a Model
Example: Train a simple CNN for 50 epochs.
```bash
python train.py --model_name simpleCNN --num_epoch 50 --batch_size 128 --lr 0.001
```

Example: Train a transfer learning model (ResNet18).
```bash
python train.py --model_name transferResNet --num_epoch 30 --batch_size 64 --lr 0.0005
```

---

## Command-Line Arguments

| Argument       | Type  | Default  | Description |
|----------------|-------|----------|-------------|
| `--exp_dir`    | str   | logs     | Directory to store TensorBoard logs |
| `--model_name` | str   | **(req)**| Model to train (`simpleCNN`, `complexCNN`, `transferResNet`, `transferMobileNet`, `transferResNetLarge`) |
| `--num_epoch`  | int   | 50       | Number of training epochs |
| `--lr`         | float | 1e-3     | Learning rate |
| `--weight_decay` | float | 1e-3   | Weight decay for optimizer |
| `--scheduler`  | bool  | False    | Use cosine annealing scheduler |
| `--seed`       | int   | 2024     | Random seed |
| `--batch_size` | int   | 256      | Batch size |

---

## Monitoring Training

To monitor training progress:
```bash
tensorboard --logdir logs
```

---

## Saving & Loading Models

- Models are automatically saved in `src/saved_models/` after training.
- To load a model:
```python
from models import load_model
model = load_model("simpleCNN", with_weights=True)
```
