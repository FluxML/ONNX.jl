import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(9, 2)
        self.linear1.weight = torch.nn.Parameter(torch.arange(18).reshape(2, 9).float())
        self.linear1.bias = torch.nn.Parameter(torch.zeros(2))
        # self.relu1 = nn.ReLU()
        # self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        y = self.linear1(x)
        # y = self.relu1(y)
        # y = self.linear2(y)
        return y


model = Net()
args = (torch.ones(1, 9),)
model(*args)
torch.onnx.export(model, args, "simple_net.onnx")