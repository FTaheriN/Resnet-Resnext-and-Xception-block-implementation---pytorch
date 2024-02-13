import torch 
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, block, num_classes=10):
        super(BaseModel, self).__init__()
        self.padding = 1
        self.in_channels = 3

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=self.padding, bias=False)
        self.bn3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2) #stride=1, padding=self.padding

        self.layer1 = self.make_layer(block, in_channels=128, out_channels=128, type=1, ch1=24, ch2=40)
        self.layer2 = self.make_layer(block, in_channels=128, out_channels=256, type=2, ch1=48, ch2=80)
        self.layer3 = self.make_layer(block, in_channels=512, out_channels=512, type=1, ch1=96, ch2=160)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(1)
        
    def make_layer(self, block, in_channels, out_channels, type, ch1, ch2):
        layers = []
        if type == 1:
            layers.append(block(in_channels, out_channels, False, ch1, ch2))
        else:
            layers.append(block(in_channels, out_channels, True, ch1, ch2))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer3(out)
        out = self.avg_pool(out)
        out = self.linear(self.dropout(self.flat(out)))
        out = self.softmax(out)
        return out