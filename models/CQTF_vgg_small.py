import torch
import torch.nn as nn
from CQTF_module import CQTF_FixedPoint_Conv2d, CQTF_FixedPoint_Linear, CQTF_FixedPoint_ConvBNFusion

class CQTF_VGG_SMALL(nn.Module):
    def __init__(self, num_classes=10):
        super(CQTF_VGG_SMALL, self).__init__()
        self.conv0 = CQTF_FixedPoint_ConvBNFusion(3, 128, kernel_size=3, padding=1, bias=False)
        self.conv1 = CQTF_FixedPoint_ConvBNFusion(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = CQTF_FixedPoint_ConvBNFusion(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv3 = CQTF_FixedPoint_ConvBNFusion(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv4 = CQTF_FixedPoint_ConvBNFusion(256, 512, kernel_size=3, padding=1, bias=False)
        self.conv5 = CQTF_FixedPoint_ConvBNFusion(512, 512, kernel_size=3, padding=1, bias=False)
        self.fc = CQTF_FixedPoint_Linear(512*4*4, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x