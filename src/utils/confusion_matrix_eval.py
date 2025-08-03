import torch
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import load_model
from src.dataloader import load_data
from src.metrics import LABEL_NAMES

def evaluate_confusion(
    model_path,
    model_name,
    data_dir="data/val",
    batch_size=64,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # load model
    model = load_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # load data
    data = load_data(data_dir, 
                        transform_pipeline="default", 
                        shuffle=False, 
                        batch_size=batch_size)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, label in data:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    # concatenate all batches
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=LABEL_NAMES)
    
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()


# testing loop
if __name__ == "__main__":
    evaluate_confusion(
        model_path=r"src\saved_models\simpleCNN.th",
        model_name="SimpleCNN",
        data_dir="data\val"
    )