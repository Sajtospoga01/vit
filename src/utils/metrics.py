import torch
from torch import nn
import pytorch_msssim

class Accuracy(nn.Module):
    """
    Defines the accuracy metric to be used in the pretraining of the model
    """

    def __init__(self):
        super().__init__()
        self.accuracy = torch.tensor(0.0)
        self.total = torch.tensor(0)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = torch.sigmoid(preds)
        # preds = (preds > 0.5).int()
        target = target.int()
        self.accuracy += (preds == target).sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.accuracy.float() / self.total


class SSIMAccuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_sum = torch.tensor(0.0)
        self.total = torch.tensor(0)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Compute SSIM between predicted and target images
        ssim_value = pytorch_msssim.ssim(
            preds,
            target,
            data_range=1.0,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            K=(0.01, 0.03)
        )
        # Update the running sum of SSIM values and the total count
        self.ssim_sum += ssim_value
        self.total += 1

    def compute(self) -> torch.Tensor:
        # Compute the average SSIM
        return self.ssim_sum.float() / self.total


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.tensor(0.0)
        self.total = torch.tensor(0)

    def update(self, loss: torch.Tensor) -> None:
        self.loss += loss
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.loss.float() / self.total


class F1Score(nn.Module):
    """
    Defines the F1 score metric to be used in the pretraining of the model
    """

    def __init__(self):
        super().__init__()
        self.f1_score = torch.tensor(0.0)
        self.total = torch.tensor(0)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).int()
        target = target.int()
        precision = (preds * target).sum() / (preds.sum() + 1e-8)
        recall = (preds * target).sum() / (target.sum() + 1e-8)
        self.f1_score += 2 * (precision * recall) / (precision + recall + 1e-8)
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.f1_score.float() / self.total