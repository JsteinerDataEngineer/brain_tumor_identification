import torch, argparse #, torchvision

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
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay)
    
    # global logger metrics
    global_step = 0
    metrics = {"train_acc": [],
                "val_acc": []}
    
    # training loop
    for epoch in range(num_epoch):
        # clear metrics
        for key in metrics:
            metrics[key].clear()

        # set model to training
        model.train()

        for img, label in train_data:
            # send data to device
            img, label = img.to(device), label.to(device)

            # forward pass
            logits = model(img)
            loss = loss_func(logits, label)

            logger.add_scalar("train/loss", loss.item(), global_step=global_step)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute accuracy
            preds = torch.argmax(logits, dim=1)
            acc = (preds == label).float().mean().item()
            metrics["train_acc"].append(acc)

            global_step += 1

        # validation loop
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                # send val data to device
                img, label = img.to(device), label.to(device)

                # compute accuracy
                preds = torch.argmax(model(img), dim=1)
                acc = (preds == label).float().mean().item()
                metrics["val_acc"].append(acc)

        # calculate accuracies at epoch level
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        # log accuracies
        logger.add_scalar("train/accuracy", epoch_train_acc, global_step=global_step)
        logger.add_scalar("val/accuracy", epoch_val_acc, global_step=global_step)

        # print first, last, or every 5th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"Training Accuracy={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
            torch.save(model.state_dict(), log_dir / f"{model_name}_epoch{epoch}.th")

    # save model
    save_model(model)

    # save a copy of model weights in log
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=256)

    # pass all arguments to the training loop
    train(**vars(parser.parse_args()))