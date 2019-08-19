from   collections import namedtuple
from   contextlib import contextmanager
from   functools import reduce, singledispatch
import math
import operator
import sys
import types

import torch
from   torch import autograd
import torch.nn as nn

# ----------------------------------------------------------------------------
# freeze

def freeze( m ):
  for p in m.parameters():
    p.requires_grad = False

# ----------------------------------------------------------------------------
# Utilities for working with tuples of tensors

class TensorTuple(tuple):
  def __new__( self, items ):
    return tuple.__new__(TensorTuple, items)
  
  @property
  def volatile( self ):
    return any( t.volatile for t in self )
    
  @volatile.setter
  def volatile( self, v ):
    for t in self: t.volatile = v
  
  def cpu( self ):
    return TensorTuple( [i.cpu() for i in self] )
    
  def numpy( self ):
    return TensorTuple( [i.numpy() for i in self] )
    
  def to( self, device ):
    return TensorTuple( [i.to( device ) for i in self] )
    
def Variable( data, *, requires_grad=False, volatile=False ):
  if isinstance(data, tuple):
    return type(data)( 
      [autograd.Variable(e, requires_grad=requires_grad, volatile=volatile)
       for e in data] )
  else:
    return autograd.Variable(data, requires_grad=requires_grad, volatile=volatile)

def foreach( f, data ):
  if isinstance(data, tuple):
    return type(data)( [f(i) for i in data] )
  else:
    return f(data)
    
def map_tuples( f, data ):
  elements = zip( *data )
  return TensorTuple( [f(e) for e in elements] )
  
def IndexSelect(base_module, index):
  def new_forward( self, *args, **kwargs ):
    ys = self._forward_impl( *args, **kwargs )
    return ys[index]
    
  base_module._forward_impl = base_module.forward
  base_module.forward = types.MethodType( new_forward, base_module )
  return base_module
  
def cat( xs, dim=0 ):
  if len(xs) > 0:
    if isinstance(xs[0], tuple):
      xs = list(zip( *xs ))
      return TensorTuple( [torch.cat( x, dim=dim ) for x in xs] )
  return torch.cat( xs, dim=dim )

# ----------------------------------------------------------------------------
# context managers

@contextmanager
def printoptions( restore={"profile": "default"}, **kwargs ):
  """ Temporarily apply `torch.set_printoptions()`. There is no way to access
  the current options, so you can provide an optional extra set of keyword args
  via `restore` to restore the desired settings on exit.
  """
  torch.set_printoptions( **kwargs )
  yield
  torch.set_printoptions( **restore )
  
@contextmanager
def printoptions_nowrap( restore={"profile": "default"}, **kwargs ):
  """ Convenience wrapper around 'printoptions' that sets 'linewidth' to its
  maximum value.
  """
  if "linewidth" in kwargs:
    raise ValueError( "user specified 'linewidth' overrides 'nowrap'" )
  torch.set_printoptions( linewidth=sys.maxsize, **kwargs )
  yield
  torch.set_printoptions( **restore )

@contextmanager
def no_grad( *variables ):
  old = [v.volatile for v in variables]
  for v in variables:
    v.volatile = True
  yield
  for v, o in zip(variables, old):
    v.volatile = o

@contextmanager
def training_mode( mode, *modules ):
  old_mode = []
  for m in modules:
    old_mode.append( m.training )
    m.train( mode )
  yield
  for (m, old) in zip(modules, old_mode):
    m.train( old )

# ----------------------------------------------------------------------------
# types

def is_batch_norm( m ):
  types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
  if isinstance(m, type):
    return issubclass( m, types )
  else:
    return isinstance( m, types )

def is_weight_layer( cls ):
  # Note: incomplete list
  types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
  return issubclass( cls, types )

# ----------------------------------------------------------------------------
# shape

def flat_size( shape ):
  return reduce( operator.mul, shape )

def _maybe_expand_tuple( dim, tuple_or_int ):
  if type(tuple_or_int) is int:
    tuple_or_int = tuple( [tuple_or_int] * dim )
  else:
    assert( type(tuple_or_int) is tuple )
  return tuple_or_int

def _output_shape_Conv( dim, input_shape, out_channels, kernel_size, stride,
                        padding, dilation, ceil_mode ):
  """ Implements output_shape for "conv-like" layers, including pooling layers.
  """
  assert( len(input_shape) == dim+1 )
  kernel_size = _maybe_expand_tuple( dim, kernel_size )
  stride      = _maybe_expand_tuple( dim, stride )
  padding     = _maybe_expand_tuple( dim, padding )
  dilation    = _maybe_expand_tuple( dim, dilation )
  quantize    = math.ceil if ceil_mode else math.floor
  out_dim = [quantize(
    (input_shape[i+1] + 2*padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) 
      / stride[i] + 1 )
    for i in range(dim)]
  output_shape = tuple( [out_channels] + out_dim )
  return output_shape

@singledispatch
def output_shape( layer, input_shape ):
  """ Computes the output shape given a layer and input shape, without
  evaluating the layer. Raises `NotImplementedError` for unsupported layer
  types.
  
  Parameters:
    `layer` : The layer whose output shape is desired
    `input_shape` : The shape of the input, in the format (N, H, W, ...). Note
      that this must *not* include a "batch" dimension or anything similar.
  """
  raise NotImplementedError( layer )

@output_shape.register(nn.BatchNorm2d)
@output_shape.register(nn.ReLU)
def _( layer, input_shape ):
  return input_shape

@output_shape.register(nn.Conv2d)
def _( layer, input_shape ):
  return _output_shape_Conv( 2, input_shape, layer.out_channels,
    layer.kernel_size, layer.stride, layer.padding, layer.dilation, False )
    
@output_shape.register(nn.Linear)
def _( layer, input_shape ):
  assert( flat_size( input_shape ) == layer.in_features )
  return tuple([layer.out_features])

@output_shape.register(nn.AvgPool2d)
def _( layer, input_shape ):
  out_channels = input_shape[0]
  return _output_shape_Conv( 2, input_shape, out_channels, layer.kernel_size,
    layer.stride, layer.padding, 1, layer.ceil_mode )

@output_shape.register(nn.MaxPool2d)
def _( layer, input_shape ):
  out_channels = input_shape[0]
  return _output_shape_Conv( 2, input_shape, out_channels, layer.kernel_size,
    layer.stride, layer.padding, layer.dilation, layer.ceil_mode )
    
@output_shape.register(nn.Sequential)
def _( layer, input_shape ):
  for m in layer.children():
    input_shape = output_shape( m, input_shape )
  return input_shape
    
# ----------------------------------------------------------------------------
# resource cost

# FIXME: It would be better to use FLOPs rather than MACCs
Flops = namedtuple( "Flops", ["macc"] )
ParameterCount = namedtuple( "ParameterCount", ["nparams", "nweights"] )

@singledispatch
def flops( layer, in_shape ):
  raise NotImplementedError( layer )

@flops.register(nn.Sequential)
def _( layer, in_shape ):
  result = Flops( 0 )
  for m in layer:
    mf = flops( m, in_shape )
    result = Flops( *(sum(x) for x in zip(mf, result)) )
    in_shape = output_shape( m, in_shape )
  return result

# FIXME: This is not correct
@flops.register(nn.BatchNorm2d)
def _( layer, in_shape ):
  return Flops( 0 )

@flops.register(nn.BatchNorm3d)
def _(layer, in_shape):
    return Flops(0)

# FIXME: This is not correct
@flops.register(nn.AvgPool2d)
def _( layer, in_shape ):
  return Flops( 0 )

@flops.register(nn.Conv2d)
def _( layer, in_shape ):
  out_shape = output_shape( layer, in_shape )
  k = reduce( operator.mul, layer.kernel_size )
  out_dim = reduce( operator.mul, out_shape )
  macc = k * in_shape[0] * out_dim / layer.groups
  return Flops( macc )
  
@flops.register(nn.Linear)
def _( layer, in_shape ):
  # assert( flat_size(in_shape) == layer.in_features )
  macc = layer.in_features * layer.out_features
  return Flops( macc )
  
@flops.register(nn.ReLU)
def _( layer, in_shape ):
  return Flops( 0 )
  
def nparams( layer ):
  n = 0
  nweights = 0
  leaf = True
  for c in layer.children():
    leaf = False
    count = nparams( c )
    n += count.nparams
    nweights += count.nweights
  if leaf:
    for p in layer.parameters():
      n += flat_size( p.data.size() )
      if is_weight_layer( layer.__class__ ):
        nweights += flat_size( p.data.size() )
  return ParameterCount( n, nweights )

# ----------------------------------------------------------------------------
# unsqueeze

def unsqueeze_right_as( x, reference ):
  while len(x.size()) < len(reference.size()):
    x = x.unsqueeze( -1 )
  return x
  
def unsqueeze_left_as( x, reference ):
  while len(x.size()) < len(reference.size()):
    x = x.unsqueeze( 0 )
  return x
  
def unsqueeze_right_to_dim( x, dim ):
  while len(x.size()) < dim:
    x = x.unsqueeze( -1 )
  return x
  
def unsqueeze_left_to_dim( x, dim ):
  while len(x.size()) < dim:
    x = x.unsqueeze( 0 )
  return x
  
# ----------------------------------------------------------------------------
# state dict

def hierarchical_state_dict( d ):
  root = dict()
  for (k, v) in d.items():
    tokens = k.split( "." )
    node = root
    for t in tokens[:-1]:
      node = node.setdefault( t, dict() )
    node[tokens[-1]] = v
  return root
  
def flat_state_dict( h ):
  flat = dict()
  def flatten( d, s=[] ):
    if isinstance(d, dict):
      for k, v in d.items():
        flatten( v, s + [k] )
    else:
      flat[".".join(s)] = d
  flatten( h )
  return flat
  
# ----------------------------------------------------------------------------
# optimizer

def optimizer_params( optimizer ):
  for group in optimizer.param_groups:
    yield from group["params"]

# ----------------------------------------------------------------------------
# Histograms
  
def histogram( x, bins ):
  N = reduce( operator.mul, x.size() )
  h = []
  s = 0
  for b in bins:
    sb = torch.sum(x <= b)
    h.append( (b, sb - s) )
    s = sb
  h.append( (math.inf, N - s) )
  return h

def pretty_histogram( h, precision=0, sep="\n" ):
  return sep.join( "{:.{prec}e}: {}".format(b, count, prec=precision) for (b, count) in h )
  
# ----------------------------------------------------------------------------
# test
  
if __name__ == "__main__":
  keys = [
    "d.bias",
    "d.w.F.0.weight",
    "d.w.F.downsample",
    "q.1.F.4.bias",
    "q.w.F.0.weight",
    "d.w.F.0.bias",
    "d.w.F.frob",
    "d.w.F.1.weight",
    "q.weight"
  ]
  
  values = list(range(len(keys)))
  d = dict( zip(keys, values) )
  print( d )
  h = hierarchical_state_dict( d )
  print( h )
  f = flat_state_dict( h )
  print( f )
