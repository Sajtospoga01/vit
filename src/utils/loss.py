from torch import nn
from torch.nn import functional as F
class CustomMSELoss(nn.Module):
    def __init__(self, penalty=20):
        super(CustomMSELoss, self).__init__()
        self.penalty = penalty

    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')

        # Apply a higher penalty if actual and predicted are on opposite sides of zero
        opposite_sides = (y_true > 0) & (y_pred < 0)
        mse_loss[opposite_sides] *= self.penalty

        # Apply different penalties based on whether the prediction is greater than the actual value
        pred_greater = y_pred > y_true
        mse_loss[pred_greater] *= self.penalty / 2

        return mse_loss.mean()