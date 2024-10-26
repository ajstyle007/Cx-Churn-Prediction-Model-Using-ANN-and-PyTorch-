from torch import nn
import torch

class CxChurn(nn.Module):

  def __init__(self):
    super().__init__()

    self.layer1 = nn.Linear(in_features=11, out_features=16)
    self.layer2 = nn.Linear(in_features=16, out_features=16)
    self.layer3 = nn.Linear(in_features=16, out_features=8)
    self.layer4 = nn.Linear(in_features=8, out_features=1)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = self.layer1(self.relu(x))
    x = self.dropout(x)
    x = self.layer2(self.relu(x))
    x = self.dropout(x)
    x = self.layer3(self.relu(x))
    x = self.dropout(x)
    x = self.layer4(x)

    return x
