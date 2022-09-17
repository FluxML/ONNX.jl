import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, A, B):
        y = torch.matmul(B, A)   # reverse order since Julia does it this way
        return y


model = Net()
args = (torch.ones(1, 9),)

import numpy as np

pyA = np.array([[1, 4.],
                [2., 5],
                [3., 6]])
pyB = np.array([
                [7., 9, 11],
                [8., 10, 12]])
args = (torch.tensor(pyA), torch.tensor(pyB))
model(*args)
torch.onnx.export(model, args, "simple_net.onnx")