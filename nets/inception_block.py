import torch 
import torch.nn as nn


class BasicConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, **kwargs),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.conv(X)

# Inception block
class InceptionBlock(nn.Module):
    """The Inception Block"""
    def __init__(self, input_channels, output_channels,
                 use_1x1conv=False, ch1=0, ch2=0, stride=1, padding=0):
        super().__init__()

        inner_channel1 = int((ch1 / 2) + (ch1 / 4))
        inner_channel2 = int((ch2 / 2) + (ch2 / 4))

        self.branch_1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConvBlock(input_channels, ch1,  kernel_size=1, stride=stride, padding=padding, bias=False)
        )
        self.branch_2 = BasicConvBlock(input_channels, ch2, kernel_size=1, stride=stride, padding=padding, bias=False)
        self.branch_3 = nn.Sequential(
            BasicConvBlock(input_channels, inner_channel1, kernel_size=1, stride=stride, padding=1, bias=False),
            BasicConvBlock(inner_channel1, ch1, kernel_size=3, stride=stride, padding=padding, bias=False),
        )
        self.branch_4 = nn.Sequential(
            BasicConvBlock(input_channels, inner_channel2, kernel_size=1, stride=stride, padding=2, bias=False),
            BasicConvBlock(inner_channel2, ch2, kernel_size=3, stride=stride, padding=padding, bias=False),
            BasicConvBlock(ch2, ch2, kernel_size=3, stride=stride, padding=padding, bias=False),
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        return out