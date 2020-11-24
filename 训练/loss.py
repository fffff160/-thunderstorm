import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        target = target.view(-1)
        loss = self.loss_func(pred, target)
        return loss

if __name__ == '__main__':
    input = torch.rand(6, 2, 256, 256).cuda()
    target = torch.ones(6, 256, 256).cuda().long()
    loss_func = MyLoss()
    loss = loss_func(input, target)
    print(loss)
