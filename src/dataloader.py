import csv
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# image labels
LABEL_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# global size, image mean, image std
SIZE = (128, 128)
MEAN = [0.186]
STD = [0.179]
RESNET_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class TumorDataset(Dataset):
    """
    Brain Tumor Dataset for classification
    """
    def __init__(
        self,
        dataset_path: str,
        transform_pipeline: str = "default"
    ):
        self.transform = self.get_transform(transform_pipeline)
        self.data = []

        dataset_path = Path(dataset_path)

        for label in LABEL_NAMES:
            label_dir = dataset_path / label
            if not label_dir.exists():
                continue
            label_id = LABEL_NAMES.index(label) 

            for img_path in label_dir.glob("*.jpg"):
                self.data.append((img_path, label_id))

    def get_transform(self, transform_pipeline: str = "default"):
        xform = None

        if transform_pipeline == "raw":
            xform = transforms.Compose([
                transforms.ToTensor()
            ])

        if transform_pipeline == "default" or transform_pipeline =="validation":
            xform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

        if transform_pipeline == "train":
            xform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize(SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

        if transform_pipeline == "pretrained_validation":
            xform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(RESNET_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

        if transform_pipeline == "pretrained_train":
            xform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.25),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        
        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified")
        
        return xform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label_id = self.data[idx]
        img = Image.open(img_path).convert("RGB") # maybe remove .convert("RGB")
        data = (self.transform(img), label_id)

        return data
    

def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 256,
    shuffle: bool = False
) -> DataLoader | Dataset:
    """
    Constructs the dataset or dataloader.

    Args:
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returnes either DataLoader or Dataset
        num_workers (int): data workers
        batch_size (int): batch size
        shuffle (bool): True for Train, False for Validation
    """
    dataset = TumorDataset(dataset_path, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return dataset
    
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )