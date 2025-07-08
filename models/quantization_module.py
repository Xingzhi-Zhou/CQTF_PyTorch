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
    
class FixedPoint_ConvBNFusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(FixedPoint_ConvBNFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if self.training:
            conv_out = self.conv(x)
            _ = self.bn(conv_out)
            batch_mean = conv_out.mean(dim=(0, 2, 3))
            batch_var = conv_out.var(dim=(0, 2, 3), unbiased=False)
        else:
            batch_mean = self.bn.running_mean
            batch_var = self.bn.running_var   

        std = torch.sqrt(batch_var + self.bn.eps)
        fused_weight = self.conv.weight * (self.bn.weight / std).reshape(-1, 1, 1, 1)

        if self.conv.bias is not None:
            fused_bias = self.bn.bias + (self.conv.bias - batch_mean) * self.bn.weight / std
        else:
            fused_bias = self.bn.bias - batch_mean * self.bn.weight / std

        qa = FixedPointQuantize.apply(x)
        qw = FixedPointQuantize.apply(fused_weight)
        qb = FixedPointQuantize.apply(fused_bias)
        output = F.conv2d(qa, qw, qb, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return output.clone()