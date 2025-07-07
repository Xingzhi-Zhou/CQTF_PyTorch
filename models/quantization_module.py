import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from config import *

class FixedPointQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.round(input / quant_step) * quant_step
        out = torch.clamp(out, min_val, max_val)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = torch.clamp(grad_input, min_val, max_val)

        return grad_input

class FixedPointConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FixedPointConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        qa = FixedPointQuantize.apply(input)
        qw = FixedPointQuantize.apply(self.weight)
        qb = FixedPointQuantize.apply(self.bias) if self.bias is not None else None
        output = F.conv2d(qa, qw, qb,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

class FixedPointLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FixedPointLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        qa = FixedPointQuantize.apply(input)
        qw = FixedPointQuantize.apply(self.weight)
        qb = FixedPointQuantize.apply(self.bias) if self.bias is not None else None
        output = F.linear(qa, qw, qb)
        return output