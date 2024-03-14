import torch 
from torch import nn

class IvertedLinearResidualBlock(nn.Module):
  def __init__(self, expand=64, squeeze=16):
    super(IvertedLinearResidualBlock, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(squeeze, expand, kernel_size=3, stride=1, padding=1, groups=squeeze, bias=False),
      nn.ReLU6(inplace=True),
      nn.Conv2d(expand, squeeze, kernel_size=1, stride=1, padding=0, bias=False),
    )
  
  def forward(self, x):
    return x + self.conv(x)
  

if __name__ == '__main__':
  x = torch.randn(10, 16, 64, 64)
  y = IvertedLinearResidualBlock(expand=64, squeeze=16)(x)
  print(y.size())
