import torch 
import torch.nn as nn
import torch.nn.functional as F


# ResnNeXt block
class ResNeXtBlock(nn.Module):
    """The ResNeXt Block"""
    def __init__(self, input_channels, output_channels,
                 use_1x1conv=False, ch1=0, ch2=0, stride=1, padding=0):
        super().__init__()

        cardinality = 32
        b_width = int(input_channels / cardinality)
        group_width = cardinality * b_width
        if use_1x1conv:
          expansion = 2
        else:
          expansion = 1

        self.conv1 = nn.Conv2d(input_channels, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*group_width)

        self.shortcut = nn.Sequential()
        if input_channels != expansion*group_width:
            self.shortcut = nn.Sequential(
              nn.Conv2d(input_channels, expansion*group_width, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(expansion*group_width)
          )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out