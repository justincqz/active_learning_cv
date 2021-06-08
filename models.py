import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self, in_c, out_c, stride=1, dropout_p=None, always_drop=False):

    super(ResidualBlock, self).__init__()
    self.p = dropout_p
    self.always_drop = always_drop

    self.main = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1), 
        nn.BatchNorm2d(out_c),
    )
    self.shortcut = nn.Sequential()
    if stride != 1 or in_c != out_c:
      self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=False)
  
  def forward(self, x):
    out = self.main(x)
    out += self.shortcut(x)
    out = F.relu(out)
    out = out if self.p is None else F.dropout(out, self.p, self.always_drop or torch.is_grad_enabled(), True)
    return out

class ResNet20(nn.Module):
  def __init__(self, in_c = 3, out_dim = 10):
    super(ResNet20, self).__init__()

    self.in_c = 16

    self.layer1 = nn.Sequential(
        nn.Conv2d(in_c, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True)
    )

    self.layer2 = self.make_layer(16, 3, 1)
    self.layer3 = self.make_layer(32, 3, 2)
    self.layer4 = self.make_layer(64, 3, 2)
    self.maxpool = nn.MaxPool2d(8)
    self.fc = nn.Linear(64, out_dim)

  def make_layer(self, out_c, num_layers, stride):
    strides = [stride] + [1] * (num_layers - 1)
    layers = []
    for s in strides:
      layers.append(ResidualBlock(self.in_c, out_c, s))
      self.in_c = out_c
    
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x