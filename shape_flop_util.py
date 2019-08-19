from functools import reduce, singledispatch
from collections import namedtuple
from contextlib import contextmanager
import math
import operator

import torch.nn as nn
from nnsearch.pytorch.gated.module import BlockGatedConv3d, BlockGatedConv2d, BlockGatedFullyConnected

Flops = namedtuple("Flops", ["macc"])

@singledispatch
def output_shape(layer, input_shape):
    """ Computes the output shape given a layer and input shape, without
    evaluating the layer. Raises `NotImplementedError` for unsupported layer
    types.

    Parameters:
        `layer` : The layer whose output shape is desired
        `input_shape` : The shape of the input, in the format (N, H, W, ...). Note
            that this must *not* include a "batch" dimension or anything similar.
    """
    raise NotImplementedError(layer)


@output_shape.register(nn.Conv2d)
def _(layer, input_shape):
    return _output_shape_Conv(2, input_shape, layer.out_channels,
                              layer.kernel_size, layer.stride, layer.padding, layer.dilation, False)

@output_shape.register(nn.Conv3d)
def _(layer, input_shape):
    return _output_shape_Conv(3, input_shape, layer.out_channels,
                              layer.kernel_size, layer.stride, layer.padding, layer.dilation, False)


@output_shape.register(nn.Linear)
def _(layer, input_shape):
    assert (flat_size(input_shape) == layer.in_features)
    return tuple([layer.out_features])


@output_shape.register(nn.AvgPool2d)
def _(layer, input_shape):
    out_channels = input_shape[0]
    return _output_shape_Conv(2, input_shape, out_channels, layer.kernel_size,
                              layer.stride, layer.padding, 1, layer.ceil_mode)
@output_shape.register(nn.Softmax)
@output_shape.register(nn.BatchNorm2d)
@output_shape.register(nn.BatchNorm3d)
@output_shape.register(nn.ReLU)
@output_shape.register(nn.Dropout)
def _(layer, input_shape):
   return input_shape

@output_shape.register(nn.MaxPool2d)
def _(layer, input_shape):
    out_channels = input_shape[0]
    return _output_shape_Conv(2, input_shape, out_channels, layer.kernel_size,
                              layer.stride, layer.padding, layer.dilation, layer.ceil_mode)

@output_shape.register(nn.MaxPool3d)
def _(layer, input_shape):
    out_channels = input_shape[0]
    return _output_shape_Conv(3, input_shape, out_channels,
                              layer.kernel_size, layer.stride, layer.padding, layer.dilation, False)

@output_shape.register(nn.Sequential)
def _(layer, input_shape):
    for m in layer.children():
        input_shape = output_shape(m, input_shape)
    return input_shape

@output_shape.register(BlockGatedFullyConnected)
def _(layer, input_shape):
    out_channels = input_shape[0]
    # print("ssss", dir(layer.components[0]).out_channel)
    out_channels = layer.components[0].out_features * len(layer.components)
    return tuple([out_channels])

@output_shape.register(BlockGatedConv3d)
def _(layer, input_shape):
    out_channels = input_shape[0]
    # print("ssss", dir(layer.components[0]).out_channel)
    out_channels= layer.components[0].out_channels * len(layer.components)
    return _output_shape_Conv(3, input_shape, out_channels,
                              layer.components[0].kernel_size, layer.components[0].stride, layer.components[0].padding, layer.components[0].dilation, False)

@singledispatch
def flops(layer, in_shape):
    raise NotImplementedError(layer)


@flops.register(nn.Sequential)
def _(layer, in_shape):
    result = Flops(0)
    for m in layer:
        mf = flops(m, in_shape)
        result = Flops(*(sum(x) for x in zip(mf, result)))
        in_shape = output_shape(m, in_shape)
    return result

@flops.register(nn.MaxPool3d)
def _(layer, in_shape):
    return Flops(0)

@flops.register(nn.BatchNorm3d)
def _(layer, in_shape):
    return Flops(0)

@flops.register(nn.Dropout)
def _(layer, in_shape):
    return Flops(0)

@flops.register(nn.Conv2d)
def _(layer, in_shape):
    out_shape = output_shape(layer, in_shape)
    k = reduce(operator.mul, layer.kernel_size)
    out_dim = reduce(operator.mul, out_shape)
    macc = k * in_shape[0] * out_dim / layer.groups
    return Flops(macc)

@flops.register(nn.Conv3d)
def _(layer, in_shape):
    out_shape = output_shape(layer, in_shape)
    k = reduce(operator.mul, layer.kernel_size)
    out_dim = reduce(operator.mul, out_shape)
    macc = k * in_shape[0] * out_dim / layer.groups
    return Flops(macc)


@flops.register(nn.Linear)
def _(layer, in_shape):
    # assert( flat_size(in_shape) == layer.in_features )
    macc = layer.in_features * layer.out_features
    return Flops(macc)


@flops.register(nn.ReLU)
def _(layer, in_shape):
    return Flops(0)


def cat(xs, dim=0):
    if len(xs) > 0:
        if isinstance(xs[0], tuple):
            xs = list(zip(*xs))
            return TensorTuple([torch.cat(x, dim=dim) for x in xs])
    return torch.cat(xs, dim=dim)


def flat_size(shape):
    return reduce(operator.mul, shape)


def _maybe_expand_tuple(dim, tuple_or_int):
    if type(tuple_or_int) is int:
        tuple_or_int = tuple([tuple_or_int] * dim)
    else:
        assert (type(tuple_or_int) is tuple)
    return tuple_or_int


def _output_shape_Conv(dim, input_shape, out_channels, kernel_size, stride,
                       padding, dilation, ceil_mode):
    """ Implements output_shape for "conv-like" layers, including pooling layers.
    """
    assert (len(input_shape) == dim + 1)
    kernel_size = _maybe_expand_tuple(dim, kernel_size)
    stride = _maybe_expand_tuple(dim, stride)
    padding = _maybe_expand_tuple(dim, padding)
    dilation = _maybe_expand_tuple(dim, dilation)
    quantize = math.ceil if ceil_mode else math.floor
    out_dim = [quantize(
        (input_shape[i + 1] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
        / stride[i] + 1)
        for i in range(dim)]
    output_shape = tuple([out_channels] + out_dim)
    return output_shape

class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        kernel_size = x.size()[2:]
        y = fn.avg_pool2d(x, kernel_size)
        while len(y.size()) > 2:
            y = y.squeeze(-1)
        return y

@output_shape.register(GlobalAvgPool2d)
def _(layer, input_shape):
    out_channels = input_shape[0]
    return (out_channels, 1, 1)


@output_shape.register(nn.MaxPool3d)
def _(layer, input_shape):
    out_channels = input_shape[0]
    return (out_channels, input_shape[1], input_shape[2])


@flops.register(GlobalAvgPool2d)
def _(layer, input_shape):
    channels = input_shape[0]
    n = input_shape[1] * input_shape[2]
    # Call a division 4 flops, minus one for the sum
    flops = channels * (n * n * 4 + (n * n - 1))
    # divide by 2 because Flops actually represents MACCs (which is stupid btw)
    return Flops(flops / 2)

@contextmanager
def printoptions_nowrap(restore={"profile": "default"}, **kwargs):
    """ Convenience wrapper around 'printoptions' that sets 'linewidth' to its
    maximum value.
    """
    if "linewidth" in kwargs:
        raise ValueError("user specified 'linewidth' overrides 'nowrap'")
    torch.set_printoptions(linewidth=sys.maxsize, **kwargs)
    yield
    torch.set_printoptions(**restore)

def unsqueeze_right_as(x, reference):
    while len(x.size()) < len(reference.size()):
        x = x.unsqueeze(-1)
    return x


def unsqueeze_left_as(x, reference):
    while len(x.size()) < len(reference.size()):
        x = x.unsqueeze(0)
    return x


def unsqueeze_right_to_dim(x, dim):
    while len(x.size()) < dim:
        x = x.unsqueeze(-1)
    return x


def unsqueeze_left_to_dim(x, dim):
    while len(x.size()) < dim:
        x = x.unsqueeze(0)
    return x
