import torch
import numpy as np


class AccuracyMetric:
    """
    Compute accuracy
    """
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        """
        Reset the metric, call before each metric
        """
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Updates using predictions and ground truth labels

        Args:
            preds (torch.Tensor): Tensor with class predictions
            labels (torch.Tensor): Tensor with ground truth class labels
        """
        self.correct += (preds.type_as(labels) == labels).sum().item()
        self.total += labels.numel()

    def compute(self) -> dict[str, float]:
        return {
            "accuracy": self.correct / (self.total + 1e-5),
            "num_samples": self.total
        }
    

class ConfusionMatrix:
    """
    Metric for computing mean IoU and Accuracy
    """
    def __init__(self, num_classes: int = 4):
        """
        Builds and updates a confusion matrix
        
        Args:
            num_classes: number of label classes
        """
        self.matrix = torch.zeros(num_classes, num_classes)
        self.class_range = torch.arange(num_classes)

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Updates using predictions and ground truth labels

        Args:
            preds (torch.Tensor): Tensor with class predictions
            labels (torch.Tensor): Tensor with ground truth class labels
        """
        if preds.dim() > 1:
            preds = preds.view(-1)
            labels = labels.view(-1)

        preds_one_hot = (preds.type_as(labels).cpu()[:, None] == self.class_range[None]).int()
        labels_one_hot = (labels.cpu()[:, None] == self.class_range[None]).int()
        update = labels_one_hot.T @ preds_one_hot

        self.matrix += update

    def reset(self):
        """
        Resets the confusion matrix, call before each epoch
        """
        self.matrix.zero_()

    def compute(self) -> dict[str, float]:
        """
        Computes mean IoU and Accuracy
        """
        true_pos = self.matrix.diagonal()
        class_iou = true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)
        mean_iou = class_iou.mean().item()
        accuracy = (true_pos.sum() / (self.matrix.sum() + 1e-5)).item()

        return {
            "iou": mean_iou,
            "accuracy": accuracy
        }