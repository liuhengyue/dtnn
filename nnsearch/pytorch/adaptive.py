import logging
import math

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as fn

log = logging.getLogger( __name__ )

# FIXME: Bug in PyTorch (https://github.com/pytorch/pytorch/issues/3397)
# prevents using infinity here -> use very big value
Inf = 1e38 # Close to maximum float32 value

class GlobalAvgPool2dGate(nn.Module):
  def __init__( self, in_features, out_features, kernel_size ):
    super().__init__()
    self.pool = nn.AvgPool2d(kernel_size)
    self.classify = nn.Linear( in_features, out_features )
    
  def forward( self, x ):
    x = self.pool( x )
    D = x.dim()
    for d in reversed(range(2, D)):
      x = x.squeeze( dim=d )
    return self.classify( x )

class differentiable_hard_gate_by_index(torch.autograd.Function):
  """ Computes a mask matrix selecting only indices less than or equal to the
  values in the input `k`. The matrix has the same shape as `H` but `H` is not
  actually used in the computation. The gradient is passed through in
  `backward()`, like the technique used in BinaryNet.
  """
  @staticmethod
  def forward( _ctx, H, k ):
    # Create mask to select unused components
    M = torch.zeros_like(H)
    mask = torch.arange(H.size(1), out=torch.LongTensor()).expand_as(H)
    keep = mask <= k # mask <= k -> always select at least one
    M[keep] = 1
    return M
    
  @staticmethod
  def backward( _ctx, grad_output ):
    # We're essentially using the BinaryNet trick of passing the gradient
    # straight through the non-differentiable function.
    return (grad_output.clone(), torch.sum(grad_output, dim=1, keepdim=True))

class differentiable_hard_gate_by_float(torch.autograd.Function):
  """ Computes a mask matrix selecting only indices less than or equal to the
  floor of the values in the input `k`, which should be floats in
  [0, ncomponents]. The matrix has the same shape as `H` but `H` is not
  actually used in the computation. The gradient is passed through in
  `backward()`, like the technique used in BinaryNet.
  """
  @staticmethod
  def forward( _ctx, H, k ):
    # Create mask to select unused components
    M = torch.zeros_like(H)
    k = torch.floor( k ).long()
    mask = torch.arange(H.size(1), out=torch.LongTensor()).expand_as(H)
    keep = mask <= k # mask <= k -> always select at least one
    M[keep] = 1
    return M
    
  @staticmethod
  def backward( _ctx, grad_output ):
    # We're essentially using the BinaryNet trick of passing the gradient
    # straight through the non-differentiable function.
    return (grad_output.clone(), torch.sum(grad_output, dim=1, keepdim=True))
    
class differentiable_threshold(torch.autograd.Function):
  @staticmethod
  def forward( _ctx, H, threshold=0, assign=-Inf ):
    G = H.clone()
    G[G <= threshold] = assign
    return G
  
  @staticmethod
  def backward( _ctx, grad_output ):
    return grad_output.clone(), None, None
    
class stochastic_binary_hard_sigmoid(torch.autograd.Function):
  @staticmethod
  def forward( ctx, logits, slope=1 ):
    ctx._slope = slope
    p = torch.clamp( 0.5 * (slope * logits + 1), 0, 1 )
    ctx._p = p
    m = torch.bernoulli( p ).type_as(logits)
    return m
  
  @staticmethod
  def backward( ctx, grad_output ):
    # Gradient is constant, only sensitive to sign
    s = torch.sign( grad_output.clone() )
    # Values that were clamped in forward() have 0 gradient
    s[(ctx._p <= 0) | (ctx._p >= 1)] = 0
    return 0.5 * ctx._slope * s, None
    
class stochastic_binary_straight_through(torch.autograd.Function):
  @staticmethod
  def forward( ctx, p, slope=1 ):
    ctx._slope = slope
    b = torch.bernoulli( p )
    return b
  
  @staticmethod
  def backward( ctx, grad_output ):
    # Gradient is constant, only sensitive to sign
    s = torch.sign( grad_output )
    return ctx._slope * s, None

class BernoulliGatedChannelStack(nn.Module):
  """ Stacks the output from an adaptive number of component modules. Gating
  decisions for each channel are made by sampling from independent Bernoulli 
  random variables.
  
  The components can be any Module instance (layers up to entire networks), as
  long as their inputs and outputs have the same shape (except for number of 
  output channels).
  """
  
  def __init__( self, in_features, components, out_shapes, Wg, initial_slope=1, normalize_output=True ):
    """
    Parameters:
      `in_features` : Number of input features
      `components` : List of component modules
      `out_shapes` : List of shapes of component outputs
      `Wg` : Module for learning the gating function
      `initial_slope` : Starting value of slope parameter for slope-annealed
        Bernoulli gradient estimator
    """
    super().__init__()
    self.ncomponents = len(components)
    self.components  = components
    self.in_features = in_features
    self.out_shapes = out_shapes
    for (i, c) in enumerate(components):
      self.add_module( "c{}".format(i), c )
    self.Wg = Wg
    self._slope = initial_slope
    self._normalize_output = normalize_output
  
  @property
  def slope( self ):
    return self._slope
  
  @slope.setter
  def slope( self, value ):
    self._slope = value
  
  def forward( self, x ):
    """ Based on SG-MoE
    """
    logits = self.Wg(x)
    p = fn.sigmoid( logits )
    G = stochastic_binary_straight_through.apply( p, self.slope )
    log.debug( "gate.G: %s", G )
    
    # Evaluate selected experts
    Ebatch = []
    for (b, xb) in enumerate(x): # each sample in batch
      xb = xb.unsqueeze( dim=0 )
      Yb = []
      active_features = 0
      for i in range(self.ncomponents):
        if G[b,i].data[0] > 0:
          # Multiply by mask value so that gradients propagate
          yi = G[b,i] * self.components[i](xb)
          Ci = yi.size(1)  # Number of output features from this component
          active_features += Ci
          Yb.append( Ci * yi ) # Weight the output for later normalization
        else:
          # Output 0's
          pad = torch.zeros( 1, *self.out_shapes[i] ).type_as(x.data)
          Yb.append( ag.Variable(pad) )
      Yb = torch.cat( Yb, dim=1 )
      if self._normalize_output and active_features > 0:
        Yb /= active_features # Normalize
      log.verbose( "gate.Gb: %s %s", b, G[b] )
      log.verbose( "gate.Yb: %s %s", b, Yb )
      Ebatch.append( Yb )
    # TODO: I'm guessing that making a list and stacking it all at once is
    # more efficient than incremental stacking. Should profile this on GPU.
    Ebatch = torch.cat( Ebatch, dim=0 )
    return Ebatch, G
    
class SoftGatedChannelStack(nn.Module):
  """ Stacks the output from an adaptive number of component modules. The
  selection is *monotonic* in the sense that component modules are selected in
  the same order until the required number have been selected; i.e. for two
  selected sets A and B, if |A| < |B| then A \subset B.
  
  The output is padded with zeros at the end to make it the same size no matter
  how many components are selected. The first component is always selected.
  
  The components can be any Module instance (layers up to entire networks), as
  long as their inputs and outputs have the same shape (except for number of 
  output channels).
  """
  
  def __init__( self, in_features, out_features, components, Wg, Wnoise ):
    """
    Parameters:
      `in_features` : Number of input features
      `out_features` : Total number of output features when all the component
        outputs are stacked.
      `components` : List of component modules
      `Wg` : Module for learning the gating function
      `Wnoise` : Module for learning the noise transformation
    """
    super().__init__()
    self.ncomponents = len(components)
    self.components  = components
    self.in_features = in_features
    self.out_features = out_features
    for (i, c) in enumerate(components):
      self.add_module( "c{}".format(i), c )
    self.Wg = Wg
    self.Wnoise = Wnoise
    
  def forward( self, x ):
    """ Based on SG-MoE
    """
    g = self.Wg(x)
    noise = ag.Variable(torch.randn(self.ncomponents)) * fn.softplus(self.Wnoise(x))
    H = g + noise
    log.debug( "gate.H: %s", H )
    
    G = differentiable_threshold.apply( H, 0, -Inf )
    G = fn.softmax( G, dim=1 )
    
    log.debug( "gate.G: %s", G )
    
    # Evaluate selected experts
    Ebatch = []
    for (b, xb) in enumerate(x): # each sample in batch
      xb = xb.unsqueeze( dim=0 )
      Yb = []
      C = 0
      size = None
      for i in range(self.ncomponents):
        if G[b,i].data[0] > 0:
          # Multiply by mask value so that gradients propagate
          yi = G[b,i] * self.components[i](xb)
          size = yi.size()
          C += yi.size(1) # Number of output channels from this component
          Yb.append( yi )
      # Fill remaining output with 0's
      short = self.out_features - C
      log.debug( "short: %s", short )
      if short > 0:
        pad = torch.zeros( 1, short, *size[2:] )
        Yb.append( ag.Variable(pad) )
      Yb = torch.cat( Yb, dim=1 )
      Ebatch.append( Yb )
    # TODO: I'm guessing that making a list and stacking it all at once is
    # more efficient than incremental stacking. Should profile this on GPU.
    Ebatch = torch.cat( Ebatch, dim=0 )
    return Ebatch, G

class MonotonicVectorGatedChannelStack(nn.Module):
  """ Stacks the output from an adaptive number of component modules. The
  selection is *monotonic* in the sense that component modules are selected in
  the same order until the required number have been selected; i.e. for two
  selected sets A and B, if |A| < |B| then A \subset B.
  
  The output is padded with zeros at the end to make it the same size no matter
  how many components are selected. The first component is always selected.
  
  The components can be any Module instance (layers up to entire networks), as
  long as their inputs and outputs have the same shape (except for number of 
  output channels).
  """
  
  def __init__( self, in_features, out_features, components, Wg, Wnoise ):
    """
    Parameters:
      `in_features` : Number of input features
      `out_features` : Total number of output features when all the component
        outputs are stacked.
      `components` : List of component modules
      `Wg` : Module for learning the gating function
      `Wnoise` : Module for learning the noise transformation
    """
    super().__init__()
    self.ncomponents = len(components)
    self.components  = components
    self.in_features = in_features
    self.out_features = out_features
    for (i, c) in enumerate(components):
      self.add_module( "c{}".format(i), c )
    self.Wg = Wg
    self.Wnoise = Wnoise
    
  def forward( self, x ):
    """ Based on SG-MoE
    """
    g = self.Wg(x)
    log.debug( "g: %s", g )
    sp = fn.softplus(self.Wnoise(x))
    log.debug( "sp: %s", sp )
    noise = ag.Variable(torch.randn(*sp.size())) * sp
    log.debug( "noise: %s", noise )
    H = g + noise
    log.debug( "H: %s", H )
    # Index of largest element = (one less than) number of components to use
    _val, k = torch.topk( H, k=1 ) # k \in {0, ..., ncomponents - 1}
    log.debug( "k: %s", k )
    G = differentiable_hard_gate_by_index.apply( H, k )
    log.debug( "sgmoe.G: %s", G )
    
    # Evaluate selected experts
    Ebatch = []
    for (b, (xb, last)) in enumerate(zip(x, k)): # each sample in batch
      xb = xb.unsqueeze( dim=0 )
      Yb = []
      C = 0
      size = None
      for i in range(last.data[0] + 1):
        # Multiply by mask value so that gradients propagate
        yi = G[b,i] * self.components[i](xb)
        size = yi.size()
        C += yi.size(1) # Number of output channels from this component
        Yb.append( yi )
      # Fill remaining output with 0's
      short = self.out_features - C
      log.debug( "short: %s", short )
      if short > 0:
        pad = torch.zeros( 1, short, *size[2:] )
        # Yb.append( ag.Variable(pad.expand( -1, -1, *size[2:] )) )
        Yb.append( ag.Variable(pad) )
      Yb = torch.cat( Yb, dim=1 )
      Ebatch.append( Yb )
    # TODO: I'm guessing that making a list and stacking it all at once is
    # more efficient than incremental stacking. Should profile this on GPU.
    Ebatch = torch.cat( Ebatch, dim=0 )
    return Ebatch, G
    
class MonotonicScalarGatedChannelStack(nn.Module):
  """ Stacks the output from an adaptive number of component modules. The
  selection is *monotonic* in the sense that component modules are selected in
  the same order until the required number have been selected; i.e. for two
  selected sets A and B, if |A| < |B| then A \subset B.
  
  The output is padded with zeros at the end to make it the same size no matter
  how many components are selected. The first component is always selected.
  
  The components can be any Module instance (layers up to entire networks), as
  long as their inputs and outputs have the same shape (except for number of 
  output channels).
  """
  
  def __init__( self, in_features, out_features, components, Wg ):
    """
    Parameters:
      `in_features` : Number of input features
      `out_features` : Total number of output features when all the component
        outputs are stacked.
      `components` : List of component modules
      `Wg` : Module for learning the gating function
    """
    super().__init__()
    self.ncomponents = len(components)
    self.components  = components
    self.in_features = in_features
    self.out_features = out_features
    for (i, c) in enumerate(components):
      self.add_module( "c{}".format(i), c )
    self.Wg = Wg
    
  def forward( self, x ):
    """ Based on SG-MoE
    """
    g = self.ncomponents * 0.5 * (1.0 + fn.tanh(self.Wg(x)))
    k = torch.floor( g )
    log.debug( "k: %s", k )
    
    G = differentiable_hard_gate_by_float.apply( ag.Variable(torch.ones(x.size(0), self.ncomponents)), g )
    log.debug( "sgmoe.G: %s", G )
    G = fn.normalize( G, p=1, dim=1 )
    
    # Evaluate selected experts
    Ebatch = []
    for (b, (xb, last)) in enumerate(zip(x, k)): # each sample in batch
      xb = xb.unsqueeze( dim=0 )
      Yb = []
      C = 0
      size = None
      for i in range(last.long().data[0] + 1):
        # Multiply by mask value so that gradients propagate
        yi = G[b,i] * self.components[i](xb)
        size = yi.size()
        C += yi.size(1) # Number of output channels from this component
        Yb.append( yi )
      # Fill remaining output with 0's
      short = self.out_features - C
      log.debug( "short: %s", short )
      if short > 0:
        pad = torch.zeros( 1, short, *size[2:] )
        # Yb.append( ag.Variable(pad.expand( -1, -1, *size[2:] )) )
        Yb.append( ag.Variable(pad) )
      Yb = torch.cat( Yb, dim=1 )
      Ebatch.append( Yb )
    # TODO: I'm guessing that making a list and stacking it all at once is
    # more efficient than incremental stacking. Should profile this on GPU.
    Ebatch = torch.cat( Ebatch, dim=0 )
    return Ebatch, G
    
class AdaptiveTopKChannelStack(nn.Module):
  """ Stacks the output from an adaptive number of component modules. The
  selection is *monotonic* in the sense that component modules are selected in
  the same order until the required number have been selected; i.e. for two
  selected sets A and B, if |A| < |B| then A \subset B.
  
  The output is padded with zeros at the end to make it the same size no matter
  how many components are selected. The first component is always selected.
  
  The components can be any Module instance (layers up to entire networks), as
  long as their inputs and outputs have the same shape (except for number of 
  output channels).
  """
  
  def __init__( self, in_features, out_features, components, Wg, Wnoise ):
    """
    Parameters:
      `in_features` : Number of input features
      `out_features` : Total number of output features when all the component
        outputs are stacked.
      `components` : List of component modules
      `Wg` : Module for learning the gating function
      `Wnoise` : Module for learning the noise transformation
    """
    super().__init__()
    self.ncomponents = len(components)
    self.components  = components
    self.in_features = in_features
    self.out_features = out_features
    for (i, c) in enumerate(components):
      self.add_module( "c{}".format(i), c )
    self.Wg = Wg
    self.Wnoise = Wnoise
    
  def forward( self, x ):
    """ Based on SG-MoE
    """
    g = self.Wg(x)
    noise = ag.Variable(torch.randn(self.ncomponents)) * fn.softplus(self.Wnoise(x))
    H = g + noise
    # Index of largest element = (one less than) number of components to use
    _val, k = torch.topk( H, k=1 ) # k \in {0, ..., ncomponents - 1}
    log.debug( "k: %s", k )
    G = differentiable_hard_gate_by_index.apply( H, k )
    log.debug( "sgmoe.G: %s", G )
    
    # Evaluate selected experts
    Ebatch = []
    for (b, (xb, last)) in enumerate(zip(x, k)): # each sample in batch
      xb = xb.unsqueeze( dim=0 )
      Yb = []
      C = 0
      size = None
      for i in range(last.data[0] + 1):
        # Multiply by mask value so that gradients propagate
        yi = G[b,i] * self.components[i](xb)
        size = yi.size()
        C += yi.size(1) # Number of output channels from this component
        Yb.append( yi )
      # Fill remaining output with 0's
      short = self.out_features - C
      log.debug( "short: %s", short )
      if short > 0:
        pad = torch.zeros( 1, short, *size[2:] )
        # Yb.append( ag.Variable(pad.expand( -1, -1, *size[2:] )) )
        Yb.append( ag.Variable(pad) )
      Yb = torch.cat( Yb, dim=1 )
      Ebatch.append( Yb )
    # TODO: I'm guessing that making a list and stacking it all at once is
    # more efficient than incremental stacking. Should profile this on GPU.
    Ebatch = torch.cat( Ebatch, dim=0 )
    return Ebatch

if __name__ == "__main__":
  # Verify same output from both implementations
  logging.basicConfig()
  logging.getLogger().setLevel( logging.DEBUG )

  print( "==== Linear layers" )
  # 42 = 
  torch.manual_seed( 542 )
  input = ag.Variable(torch.randn( 4, 3 ))
  fake_loss = torch.randn( 4, 2*3 )
  components = [nn.Linear(3, 2), nn.Linear(3, 2), nn.Linear(3, 2)]
  Wg = nn.Linear( 3, len(components) )
  Wnoise = nn.Linear( 3, len(components) )
  
  # stack = MonotonicVectorGatedChannelStack( 3, 2*len(components), components, Wg, Wnoise )
  stack = SoftGatedChannelStack( 3, 2*len(components), components, Wg, Wnoise )
  
  torch.manual_seed( 314 )
  output = stack(input)
  print( output )
  output.backward( fake_loss )
  for c in stack.components:
    print( c )
    if c.weight.grad is not None:
      print( c.weight.grad.data )
    else:
      print( "No grad" )
  print( stack.Wg )
  if stack.Wg.weight.grad is not None:
    print( stack.Wg.weight.grad.data )
  else:
    print( "No grad" )
  print( stack.Wnoise )
  if stack.Wnoise.weight.grad is not None:
    print( stack.Wnoise.weight.grad.data )
  else:
    print( "No grad" )
  for p in stack.parameters():
    if p.grad is not None:
      p.grad.data.zero_()
      
  # import sys
  # sys.exit(0)
  
  print( "==== Conv layers" )
  # 42 = 
  torch.manual_seed( 48 )
  input = ag.Variable(torch.randn( 4, 3, 9, 9 ))
  fake_loss = torch.randn( 4, 2*3, 7, 7 )
  components = [nn.Conv2d(3, 2, 3), nn.Conv2d(3, 2, 3), nn.Conv2d(3, 2, 3)]
  Wg = GlobalAvgPool2dGate( 3, len(components), 9 )
  Wnoise = GlobalAvgPool2dGate( 3, len(components), 9 )
  
  # stack = MonotonicVectorGatedChannelStack( 3, 2*len(components), components, Wg, Wnoise )
  stack = SoftGatedChannelStack( 3, 2*len(components), components, Wg, Wnoise )
  
  torch.manual_seed( 314 )
  output = stack(input)
  print( output )
  output.backward( fake_loss )
  for c in stack.components:
    print( c )
    if c.weight.grad is not None:
      print( c.weight.grad.data )
    else:
      print( "No grad" )
  print( stack.Wg )
  if stack.Wg.classify.weight.grad is not None:
    print( stack.Wg.classify.weight.grad.data )
  else:
    print( "No grad" )
  print( stack.Wnoise )
  if stack.Wnoise.classify.weight.grad is not None:
    print( stack.Wnoise.classify.weight.grad.data )
  else:
    print( "No grad" )
  for p in stack.parameters():
    if p.grad is not None:
      p.grad.data.zero_()
      
  import sys
  sys.exit(0)
  
  print( "==== Scalar: Linear layers" )
  # 42 = 
  torch.manual_seed( 51 )
  input = ag.Variable(torch.randn( 4, 3 ))
  fake_loss = torch.randn( 4, 2*3 )
  components = [nn.Linear(3, 2), nn.Linear(3, 2), nn.Linear(3, 2)]
  Wg = nn.Linear( 3, 1 )
  Wnoise = nn.Linear( 3, len(components) )
  
  stack = MonotonicVectorGatedChannelStack( 3, 2*len(components), components, Wg, Wnoise )
  
  torch.manual_seed( 314 )
  output = stack(input)
  print( output )
  output.backward( fake_loss )
  for c in stack.components:
    print( c )
    if c.weight.grad is not None:
      print( c.weight.grad.data )
    else:
      print( "No grad" )
  print( stack.Wg )
  if stack.Wg.weight.grad is not None:
    print( stack.Wg.weight.grad.data )
  else:
    print( "No grad" )
  for p in stack.parameters():
    if p.grad is not None:
      p.grad.data.zero_()
  
  print( "==== Scalar: Conv layers" )
  # 42 = 
  torch.manual_seed( 48 )
  input = ag.Variable(torch.randn( 4, 3, 9, 9 ))
  fake_loss = torch.randn( 4, 2*3, 7, 7 )
  components = [nn.Conv2d(3, 2, 3), nn.Conv2d(3, 2, 3), nn.Conv2d(3, 2, 3)]
  Wg = GlobalAvgPool2dGate( 3, 1, 9 )
  Wnoise = GlobalAvgPool2dGate( 3, len(components), 9 )
  
  stack = MonotonicScalarGatedChannelStack( 3, 2*len(components), components, Wg )
  
  torch.manual_seed( 314 )
  output = stack(input)
  print( output )
  output.backward( fake_loss )
  for c in stack.components:
    print( c )
    if c.weight.grad is not None:
      print( c.weight.grad.data )
    else:
      print( "No grad" )
  print( stack.Wg )
  if stack.Wg.classify.weight.grad is not None:
    print( stack.Wg.classify.weight.grad.data )
  else:
    print( "No grad" )
  for p in stack.parameters():
    if p.grad is not None:
      p.grad.data.zero_()
