import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

from quantization_module import FixedPointQuantize

from config import *

@triton.jit
def CQTF_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SCALE,  # Quantization scale factor
    MAX_VAL,  # Maximum value for quantization
    MIN_VAL  # Minimum value for quantization
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = B_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + tl.arange(0, BLOCK_K)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + tl.arange(0, BLOCK_K)[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        dot_result = tl.dot(a, b)
        # dot_result = round_func(dot_result / SCALE) * SCALE
        dot_result = tl.floor(dot_result / SCALE + 0.5) * SCALE
        dot_result = tl.clamp(dot_result, MIN_VAL, MAX_VAL)

        acc += dot_result
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Quantize the accumulator
    acc = tl.clamp(acc, MIN_VAL, MAX_VAL)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def CQTF_matmul(A, B):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    grid = lambda META: (
        (M + META['BLOCK_M'] - 1) // META['BLOCK_M'],
        (N + META['BLOCK_N'] - 1) // META['BLOCK_N']
    )

    CQTF_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
        SCALE=quant_step, MAX_VAL=max_val, MIN_VAL=min_val
    )
    return C

def CQTF_linear(input, weight, bias=None):
    output = CQTF_matmul(input, weight.T)
    if bias is not None:
        output += bias
    return output

class CQTF_LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight)

        output = CQTF_linear(input, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = torch.matmul(grad_output, weight)

        grad_weight = torch.matmul(grad_output.T, input)

        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias
    
def CQTF_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape

    if isinstance(padding, tuple):
        padding_height, padding_width = padding
    else:
        padding_height = padding_width = padding

    if isinstance(stride, tuple):
        stride_height, stride_width = stride
    else:
        stride_height = stride_width = stride

    if isinstance(dilation, tuple):
        dilation_height, dilation_width = dilation
    else:
        dilation_height = dilation_width = dilation

    input_unfolded = F.unfold(input, kernel_size=(kernel_height, kernel_width),
                              stride=(stride_height, stride_width),
                              padding=(padding_height, padding_width),
                              dilation=(dilation_height, dilation_width))
    input_unfolded = input_unfolded.transpose(1, 2).reshape(-1, input_unfolded.size(1))

    weight_unfolded = weight.view(out_channels, -1).T

    output_unfolded = CQTF_matmul(input_unfolded, weight_unfolded)

    if bias is not None:
        output_unfolded += bias.view(1, -1)

    out_height = (in_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    out_width = (in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1

    output = output_unfolded.view(batch_size, out_height, out_width, out_channels).permute(0, 3, 1, 2)
    return output
    
class CQTF_Conv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.bias = bias

        output = CQTF_conv2d(input, weight, bias, stride, padding, dilation, groups)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        bias = ctx.bias

        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)

        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None
    
class CQTF_FixedPoint_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CQTF_FixedPoint_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        qa = FixedPointQuantize.apply(input)
        qw = FixedPointQuantize.apply(self.weight)
        qb = FixedPointQuantize.apply(self.bias) if self.bias is not None else None
        output = CQTF_Conv2dFunction.apply(qa, qw, qb,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

class CQTF_FixedPoint_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CQTF_FixedPoint_Linear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        qa = FixedPointQuantize.apply(input)
        qw = FixedPointQuantize.apply(self.weight)
        qb = FixedPointQuantize.apply(self.bias) if self.bias is not None else None
        output = CQTF_LinearFunction.apply(qa, qw, qb)
        return output
    
class CQTF_FixedPoint_ConvBNFusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(CQTF_FixedPoint_ConvBNFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        conv_out = self.conv(x)

        if self.training:
            _ = self.bn(conv_out)
            batch_mean = conv_out.mean(dim=(0, 2, 3))
            batch_var = conv_out.var(dim=(0, 2, 3), unbiased=False)
        else:
            batch_mean = self.bn.running_mean
            batch_var = self.bn.running_var

        std = torch.sqrt(batch_var + self.bn.eps)
        fused_weight = self.conv.weight * (self.bn.weight / std).reshape(-1, 1, 1, 1)
        fused_bias = self.bn.bias - batch_mean * self.bn.weight / std

        qa = FixedPointQuantize.apply(x)
        qw = FixedPointQuantize.apply(fused_weight)
        qb = FixedPointQuantize.apply(fused_bias)
        output = CQTF_Conv2dFunction.apply(qa, qw, qb, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return output.clone()