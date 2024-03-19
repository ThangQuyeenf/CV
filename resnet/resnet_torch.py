import torch
from torch import nn
from torchsummary import summary
from residual_torch import ResidualBlockTorch

class ResNetTorch(nn.Module):
  def __init__(self, blocks, num_classes=10):
    super(ResNetTorch, self).__init__()
    self.inplanes = 64
    self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.residual_blocks = nn.Sequential(*blocks)
    self.avgpool = nn.Flatten()
    self.dense = nn.Linear(in_features=512, out_features=num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.residual_blocks(x)
    x = self.avgpool(x)
    x = self.dense(x)
    return x


blocks = [
  ResidualBlockTorch(64, 64, 2),
  ResidualBlockTorch(64, 64, 2),
  ResidualBlockTorch(64, 128, 2),
  ResidualBlockTorch(128, 128, 2),
  ResidualBlockTorch(128, 256, 2),
  ResidualBlockTorch(256, 256, 2),
  ResidualBlockTorch(256, 512, 2),
  ResidualBlockTorch(512, 512, 2),
]

if __name__ == '__main__':
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model = ResNetTorch(blocks, num_classes=10)
 summary(model, (1, 28, 28))
