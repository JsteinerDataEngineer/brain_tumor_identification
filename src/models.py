from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DIRECTORY = Path("src") / "saved_models"
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # CNN block
        self.conv_block = nn.Sequential(                    # input: 3x128x128
            nn.Conv2d(3, 16, kernel_size=3, padding=1),     # 16x128x128
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 16x64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # 32x64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 32x32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 64x32x32
            nn.ReLU(),
            nn.MaxPool2d(2)                                 # 64x16x16
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    # forward pass
    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x



MODEL_FACTORY = {
    "simpleCNN": SimpleCNN
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs
) -> torch.nn.Module:
    """
    Load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = DIRECTORY / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"

            ) from e
        
    return m
        

def save_model(model: torch.nn.Module) -> str:
    """
    Function used to save model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")
    
    output_path = DIRECTORY / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path