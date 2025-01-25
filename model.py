import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer_1 = nn.Linear(in_features=1, out_features=3)
        self.layer_2 = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
