import torch, torchvision, argparse

import numpy as np
import torch.utils.tensorboard as tb

from datetime import datetime
from pathlib import Path

from .models import load_model, save_model
from .dataloader import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "CNN",
    num_epoch: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 256,
    seed: int = 2024,
    **kwargs
):
    # use GPU or MPS if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create directory and tensorboard
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # call and log model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # load data
    train_data = load_data("data/train",
                           transform_pipeline="default",
                           shuffle=True,
                           batch_size=batch_size,
                           num_workers=4)
    val_data = load_data("data/val",
                         transform_pipeline="default",
                         shuffle=False)
    
    logger.add_images("train_images", next(iter(train_data))[0])
    logger.add_images("val_images", next(iter(val_data))[0])

    # create loss function and optimizer
    ##### Need to add loss
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay)
    
    ## need to add training loop