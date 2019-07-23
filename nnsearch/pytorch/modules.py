import logging
import types

import torch
from   torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as fn

from   nnsearch.pytorch import torchx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------
# utility

def ScheduledParameters( base_type, **schedules ):
  """ Derives a new class from `base_type` that updates its attributes to
  track the value of specified `Hyperparameter`s. Returns a type that inherits
  from `base_type` and that before every call to `forward()`, executes
  `setattr(self, key, value())` for every `(key, value)` pair in `schedules`.
  
  The reason for deriving from the base_type rather than simply wrapping it is
  to avoid changing the names of Parameters for PyTorch save/load.
  """
  if not issubclass(base_type, nn.Module):
    raise ValueError( "base_type must be a subclass of nn.Module" )
  
  class ScheduledParametersClass(base_type):
    def __init__( self, *args, **kwargs ):
      super().__init__( *args, **kwargs )
      self.__schedules = schedules
      for name in schedules:
        if not hasattr(self, name):
          raise ValueError( "attribute does not exist: {}".format(name) )
      
    def forward( self, *args, **kwargs ):
      for name, value in self.__schedules.items():
        setattr( self, name, value() )
        log.debug( "__dict__: %s", self.__dict__ )
        log.debug( "ScheduledParameters: %s = %s <- %s",
                   name, getattr( self, name ), value )
      return super().forward( *args, **kwargs )
      
  return ScheduledParametersClass
  
def FrozenBatchNorm( base_module ):
  """ Patches a module instance so that all `BatchNorm` child modules
    (1) have `requires_grad = False` in their weights
    (2) are always in "eval" mode
    
  Changes `base_module` in-place and returns `base_module`.
  """
  def train( self, mode=True ):
    super(type(base_module), self).train( mode )
    if mode:
      for m in self.modules():
        if torchx.is_batch_norm( m ):
          m.train( False )
  
  base_module.train = types.MethodType( train, base_module )
  for m in base_module.modules():
    if torchx.is_batch_norm( m ) and m.affine:
      m.weight.requires_grad = False
      m.bias.requires_grad = False
  # Freezing doesn't take effect until train() is called
  base_module.train( base_module.training )
  return base_module

# ----------------------------------------------------------------------------
# linear

class FullyConnected(nn.Linear):
  """ A fully-connected layer. Flattens its input automatically.
  """
  def __init__( self, *args, **kwargs ):
    super().__init__( *args, **kwargs )
    
  def forward( self, x ):
    n = x.size(0)
    flat = x.view(n, -1)
    return super().forward( flat )

class MultiLayerPerceptron(nn.Sequential):
  def __init__( self, in_channels, nunits, activation=nn.ReLU, dropout=0, 
                batch_norm=False ):
    assert( in_channels > 0 )
    assert( len(nunits) > 0 )
    assert( 0 <= dropout < 1 )
    layers = []
    for n in nunits:
      layers.append( FullyConnected( in_channels, n ) )
      if batch_norm:
        layers.append( nn.BatchNorm1d( n ) )
      layers.append( activation() )
      if dropout > 0:
        layers.append( nn.Dropout( dropout ) )
      in_channels = n
    super().__init__( *layers )
    
# ----------------------------------------------------------------------------
# activation

class BoundedSigmoid(nn.Module):
  def __init__( self, alpha ):
    super().__init__()
    self.alpha = alpha
    
  def forward( self, x ):
    x = fn.sigmoid( x )
    x = self.alpha * x + (1.0 - self.alpha) * (1.0 - x)
    return x
    
  def __repr__( self ):
    return "BoundedSigmoid({})".format( self.alpha )
    
class Rescale(nn.Module):
  def __init__( self, in_range, out_range ):
    assert in_range[0] < in_range[1]
    assert out_range[0] < out_range[1]
    super().__init__()
    self.in_range = in_range
    in_spread = in_range[1] - in_range[0]
    self.out_range = out_range
    out_spread = out_range[1] - out_range[0]
    self.scale = out_spread / in_spread
    self.bias = self.out_range[0] - self.scale*self.in_range[0]
    # Avoid extreme weirdness with np.floatX types:
    # https://github.com/pytorch/pytorch/issues/4433
    assert type(self.scale) == float
    assert type(self.bias)  == float
    
  def forward( self, x ):
    return self.scale*x + self.bias
    
class _clamp_invert_gradient(Function):
  @staticmethod
  def forward( ctx, x, min, max ):
    ctx.save_for_backward( x )
    ctx._min = min
    ctx._max = max
    # return torch.clamp( x, min=min, max=max )
    return torch.max(torch.min(x, max), min)
    
  @staticmethod
  def backward( ctx, grad_output ):
    p, = ctx.saved_variables
    min = ctx._min
    max = ctx._max
    # Masks for positive and negative gradients
    # Note that since PyTorch does gradient *descent*, a positive gradient will
    # result in *decreasing* the parameter.
    # Note that 0's will be preserved since we later multiply by `grad_output`
    dec = torch.sign( grad_output )
    dec[dec < 0] = 0 # Positive gradient -> decrease parameter
    inc = 1 - dec    # Otherwise increase parameter
    # Masked gradient scaling factors
    pinc = ((max - p) / (max - min)) * inc
    pdec = ((p - min) / (max - min)) * dec
    return grad_output * (pinc + pdec), None, None
    
def clamp_invert_gradient( x, min, max ):
  """ Same as `torch.clamp` in the forward pass. In the backward pass, scales
  down gradients as the corresponding outputs approach the clamp limits, and
  reverses the gradient sign if the output exceeds the limit.
  
  @article{hausknecht2016deep,
    title={Deep reinforcement learning in parameterized action space},
    author={Hausknecht, Matthew and Stone, Peter},
    journal={arXiv preprint arXiv:1511.04143},
    year={2016}
  }
  """
  return _clamp_invert_gradient.apply(x, min, max)
  
class ClampInvertGradient(nn.Module):
  def __init__( self, min, max ):
    super().__init__()
    self.min = min
    self.max = max
    
  def forward( self, x ): 
    return clamp_invert_gradient( x, self.min, self.max )
    
# ----------------------------------------------------------------------------
# functions

class Identity(nn.Module):
  def forward( self, x ):
    return x
    
@torchx.output_shape.register(Identity)
def _( layer, input_shape ):
  return input_shape
  
@torchx.flops.register(Identity)
def _( layer, input_shape ):
  return torchx.Flops( 0 )

# ----------------------------------------------------------------------------

class Lambda(nn.Module):
  def __init__( self, f ):
    super().__init__()
    self.f = f
    
  def forward( self, x ):
    return self.f( x )

class Log(nn.Module):
  def forward( self, x ):
    return torch.log( x )

# ----------------------------------------------------------------------------
# global pooling
    
class GlobalAvgPool2d(nn.Module):
  def forward( self, x ):
    kernel_size = x.size()[2:]
    y = fn.avg_pool2d( x, kernel_size )
    while len(y.size()) > 2:
      y = y.squeeze(-1)
    return y
    
@torchx.output_shape.register(GlobalAvgPool2d)
def _( layer, input_shape ):
  out_channels = input_shape[0]
  return (out_channels, 1, 1)
  
@torchx.flops.register(GlobalAvgPool2d)
def _( layer, input_shape ):
  channels = input_shape[0]
  n = input_shape[1] * input_shape[2]
  # Call a division 4 flops, minus one for the sum
  flops = channels * (n*n*4 + (n*n - 1))
  # divide by 2 because Flops actually represents MACCs (which is stupid btw)
  return torchx.Flops( flops / 2 )
