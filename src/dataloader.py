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
MEAN = [0.2788, 0.2657, 0.2629]
STD = [0.2064, 0.1944, 0.2252]


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

        if transform_pipeline == "default":
            xform = transforms.Compose([
                transforms.Resize(SIZE),
                transforms.ToTensor()
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