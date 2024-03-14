import torch
from torch import nn

class DepthwiseSeparableConv(nn.Module):
  def __init__(self, nin, nout):
    super(DepthwiseSeparableConv, self).__init__()
    self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
    self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

  def forward(self, x):
      out = self.depthwise(x)
      out = self.pointwise(out)
      return out


x = torch.randn(10, 20, 15, 15) # Batch_size, C, W, H
y = DepthwiseSeparableConv(nin=20, nout=50)(x)
print(y.size())