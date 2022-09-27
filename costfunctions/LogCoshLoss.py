''' 
Numerically stable Log_cosh loss function (lacking in Pytorch).

Implementation from:
    https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch. 

It works the same way as the Keras implementation:
    https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617
'''

import math
import torch
import torch.nn as nn

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + nn.functional.softplus(-2. * x) - math.log(2.0)
    
    return _log_cosh(y_pred - y_true)

class LogCoshLoss(nn.Module):
    
    def __init__(self, reduction : str = 'mean'):
        super().__init__()
        if reduction in ['mean', 'sum', 'none']:
            self.reduction = reduction
        else:
            raise NotImplementedError(
                f"Reduction method {reduction} not\
                implemented, please try \
                'mean', 'sum', or 'none'.")


    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        loss = log_cosh_loss(y_pred, y_true)
        if self.reduction == 'mean': 
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
