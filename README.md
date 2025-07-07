# Computation-Quantized Training Framework with PyTorch
A Computation-Quantized Training Framework to Generate Accuracy Lossless QNNs for Embedded Systems

## Training QNNs
```bash
$ cd ./training
```
An example (cifar10_CQTF_VGG_SMALL.py) can be referenced.
```bash
$ python cifar10_CQTF_VGG_SMALL.py
```

## The Quantization-Mode of Computation
- RC : ceil 
- RN : round 
- RF : floor

Need to manually change the quantization mode in CQTF_matmul_kernel
```python
QUANT_MODE = 'RN'
# QUANT_MODE = 'RC'
# QUANT_MODE = 'RF'

# Need to manually change the quantization mode in CQTF_matmul_kernel
# Line 42 of .\models\CQTF_module.py
if QUANT_MODE == 'RN':
    round_func = lambda x: tl.floor(x + 0.5)
elif QUANT_MODE == 'RC':
    round_func = tl.ceil
elif QUANT_MODE == 'RF':
    round_func = tl.floor  
```

## QNN Examples
Build the QNN with CQTF_FixedPoint_Conv2d, CQTF_FixedPoint_Linear and CQTF_FixedPoint_ConvBNFusion.
```python
import torch
import torch.nn as nn
from CQTF_module import CQTF_FixedPoint_Conv2d, CQTF_FixedPoint_Linear, CQTF_FixedPoint_ConvBNFusion

class CQTF_VGG_SMALL(nn.Module):
    def __init__(self, num_classes=10):
        super(CQTF_VGG_SMALL, self).__init__()
        self.conv0 = CQTF_FixedPoint_ConvBNFusion(3, 128, kernel_size=3, padding=1, bias=False)
        self.conv1 = CQTF_FixedPoint_ConvBNFusion(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
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
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
```