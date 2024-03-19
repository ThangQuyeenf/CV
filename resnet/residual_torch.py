import torch
from torch import nn

class ResidualBlockTorch(nn.Module):
  def __init__(self, in_channels, out_channels, strides=1):
    super(ResidualBlockTorch, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = nn.Sequential()
    if strides!= 1 or in_channels!= out_channels:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.downsample = nn.Sequential()
    
  
  def forward(self, x):
    residual = self.downsample(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    out = self.relu(out)
    return out

    

if __name__ == '__main__':
  X = torch.rand((4, 1, 28, 28))
  Y = ResidualBlockTorch(in_channels=1, out_channels=64, strides=2)(X)
  print(Y.size())