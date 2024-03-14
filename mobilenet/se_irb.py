import torch 
from torch import nn

class IvertedLinearResidualBlock(nn.Module):
  def __init__(self, expand=64, squeeze=16):
    super(IvertedLinearResidualBlock, self).__init__()
    self.residual = nn.Sequential(
      nn.Conv2d(squeeze, expand, kernel_size=3, stride=1, padding=1, groups=squeeze, bias=False),
      nn.ReLU6(inplace=True),
      nn.Conv2d(expand, squeeze, kernel_size=1, stride=1, padding=0, bias=False),
    )
    self.se = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(expand, squeeze,kernel_size=1, stride=1, padding=0, groups=squeeze, bias=False),
      nn.ReLU6(inplace=True),
      nn.Conv2d(squeeze, expand,kernel_size=1, stride=1, padding=0, bias=False),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    return x + self.residual(x) + x*self.se(x)


class SEBlock(nn.Module):
  def __init__(self, channel, reduction=4):
    super(SEBlock, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channel // reduction, channel, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y

class SEResidualBlock(nn.Module):
  def __init__(self, in_c, out_c, stride=1):
    super(SEResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_c)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, groups=out_c, bias=False)
    self.bn2 = nn.BatchNorm2d(out_c)
    self.se = SEBlock(out_c)
    self.shortcut = nn.Sequential()
    if stride!= 1 or in_c!= out_c:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_c)
      )

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.se(out)
    out += self.shortcut(x)
    out = self.relu(out)
    return out

if __name__ == '__main__':
  x = torch.randn(10, 16, 64, 64)
  y = SEResidualBlock(in_c=16, out_c=16)(x)
  print(y.size())
