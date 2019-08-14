from torch import nn
from nnsearch.pytorch.gated.module import (GatedConcat, BlockGatedConv2d, BlockGatedConv3d,
                                           BlockGatedFullyConnected, GatedChainNetwork)

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def gated_conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True,
               dilation=1, stride=1, relu=True, bias=True, ncomponents=4):
    modules = [BlockGatedConv2d(ncomponents, in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def gatedConvBlock(ncomponents, in_shape, in_channels, out_channels, kernel_size=3, padding=1, bn=True,
              dilation=1, stride=1, relu=True, bias=True):
    modules = [BlockGatedConv2d(ncomponents, in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, bias=bias),]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU())

    # set gated modules
    # compute output shape
    s = (in_shape[1] - kernel_size + 2 * padding) // stride + 1
    out_shape = (out_channels, s, s)
    gated_modules = [(modules[0], (in_channels,) + in_shape[1:])]
    return modules, gated_modules, out_shape

def gatedDwConvBlock(ncomponents, in_shape, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    modules = [
        BlockGatedConv2d(ncomponents, in_channels, in_channels, kernel_size, stride=stride, 
            padding=padding, dilation=dilation, groups=in_channels//ncomponents, bias=True), 
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),

        BlockGatedConv2d(ncomponents, in_channels, out_channels, 1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()]

    # set gated modules
    # compute output shape
    s = (in_shape[1] - kernel_size + 2 * padding) // stride + 1
    out_shape = (in_channels, s, s)
    gated_modules = [(modules[0], (in_channels,) + in_shape[1:]), (modules[3], out_shape)]
    return modules, gated_modules, out_shape

def gated3dConvBlock(ncomponents, in_shape, in_channels, out_channels, kernel_size=3, padding=1, bn=True,
              dilation=1, stride=1, relu=True, bias=True):
    modules = [BlockGatedConv3d(ncomponents, in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation, bias=bias),]
    if bn:
        modules.append(nn.BatchNorm3d(out_channels))
    if relu:
        modules.append(nn.ReLU())

    # set gated modules
    # compute output shape
    # convert pars to tuple if nessary
    padding = tuple([padding] * 3) if type(padding) == int else padding
    dilation = tuple([dilation] * 3) if type(dilation) == int else dilation
    kernel_size = tuple([kernel_size] * 3) if type(kernel_size) == int else kernel_size
    stride = tuple([stride] * 3) if type(stride) == int else stride
    c, d, h, w = in_shape
    d = (d + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    h = (h + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    w = (w + 2*padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1
    out_shape = (out_channels, d, h, w)
    gated_modules = [(modules[0], (in_channels,) + in_shape[1:])]
    return modules, gated_modules, out_shape

def Maxpool3dWrapper(in_shape, kernel_size=2, stride=1, padding=0, dilation=1):
    modules = [nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)]

    # set gated modules
    # compute output shape
    # convert pars to tuple if nessary
    padding = tuple([padding] * 3) if type(padding) == int else padding
    dilation = tuple([dilation] * 3) if type(dilation) == int else dilation
    kernel_size = tuple([kernel_size] * 3) if type(kernel_size) == int else kernel_size
    stride = tuple([stride] * 3) if type(stride) == int else stride
    c, d, h, w = in_shape
    d = (d + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    h = (h + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    w = (w + 2*padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1
    out_shape = (c, d, h, w)
    return modules, out_shape