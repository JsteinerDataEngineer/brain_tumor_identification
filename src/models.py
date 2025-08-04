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
        self.conv_block = nn.Sequential(                    # input: 1x128x128
            nn.Conv2d(1, 16, kernel_size=3, padding=1),     # 16x128x128
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

class ComplexCNN(nn.Module):
    class Block(nn.Module):
        def __init__(self,
                     in_channels: int,
                     out_channels: int,
                     stride: int,
                     kernel_size: int = 3):
            super().__init__()

            padding = (kernel_size - 1) // 2

            self.c1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.n1 = nn.GroupNorm(1, out_channels)
            self.relu1 = nn.ReLU()
            
            self.c2 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
            self.n2 = nn.GroupNorm(1, out_channels)
            self.relu2 = nn.ReLU()

            self.skip = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0
            ) if in_channels != out_channels else nn.Identity()

        def forward(self, xi):
            x = self.relu1(self.n1(self.c1(xi)))
            x = self.relu2(self.n2(self.c2(x)))
            return self.skip(xi) + x
        
    def __init__(self,
                 in_channels: int = 1,
                 channels_l0: int = 32,
                 n_blocks: int = 3,
                 num_classes: int = 6):
        super().__init__()

        cnn_layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels_l0,
                kernel_size=11,
                stride=2,
                padding=5
            ),
            nn.ReLU()
        ]

        c1 = channels_l0

        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(in_channels=c1, out_channels=c2, stride=2))
            c1 = c2

        cnn_layers.append(nn.Conv2d(c1, num_classes, kernel_size=1))

        self.network = nn.Sequential(*cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).mean(dim=-1).mean(dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(dim=1)


MODEL_FACTORY = {
    "simpleCNN": SimpleCNN,
    "complexCNN": ComplexCNN
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