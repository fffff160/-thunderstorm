import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(weight = torch.Tensor((1,500)).cuda())

    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        target = target.view(-1)
        loss = self.loss_func(pred, target)
        return loss
