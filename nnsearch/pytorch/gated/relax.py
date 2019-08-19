import logging
import math

import numpy as np

import torch
from   torch.autograd import Function, Variable
import torch.nn.functional as fn

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------
# Tensor creation

class gate_matrix_from_count(Function):
  @staticmethod
  def forward( ctx, c, n ):
    if c.size() == torch.Size([]):
      batch_size = 1
    else:
      batch_size = c.size(0) # sometime error for last batch
    g = torch.arange(1, n+1).expand(batch_size, n).type_as(c) # Each row == [1,...,n]
    p = c.unsqueeze(-1).expand(batch_size, n)      # Each column == c
    g = (g <= p).type_as(c) # Convert to [1 1 ... 1 0 0 ... 0] numeric
    return g
    
  @staticmethod
  def backward( ctx, grad_output ):
    cgrad = torch.sum( grad_output, dim=1 )
    return cgrad, None

# ----------------------------------------------------------------------------
# Relaxations of deterministic discretizations

class straight_through(Function):
  @staticmethod
  def forward( ctx, f, x ):
    return f( x )
    
  @staticmethod
  def backward( ctx, grad_output ):
    return None, grad_output

class threshold_straight_through(Function):
  @staticmethod
  def forward( ctx, x, clip_grad=None ):
    if clip_grad is not None:
      assert( clip_grad >= 0 )
    ctx._clip_grad = clip_grad
    return (x > 0).type_as(x)
    
  @staticmethod
  def backward( ctx, grad_output ):
    s = grad_output.clone()
    if ctx._clip_grad is not None:
      # s[torch.abs(s) > ctx._clip_grad] = 0
      s = torch.clamp( s, min=-ctx._clip_grad, max=ctx._clip_grad )
    return s, None, None
    
class threshold_straight_through_sign(Function):
  @staticmethod
  def forward( ctx, x, slope=1 ):
    ctx._slope = slope
    return (x > 0).type_as(x)
    
  @staticmethod
  def backward( ctx, grad_output ):
    # Gradient is constant, only sensitive to sign
    s = torch.sign( grad_output )
    return ctx._slope * s, None, None

# ----------------------------------------------------------------------------
# Relaxations of discrete sampling

class bernoulli_straight_through(Function):
  """ Samples from a bernoulli distribution in `forward()` and passes the
  gradient straight through in `backward()`.
  """

  @staticmethod
  def forward( ctx, p, clip_grad=None ):
    if clip_grad is not None:
      assert( clip_grad >= 0 )
    ctx._clip_grad = clip_grad
    b = torch.bernoulli( p )
    return b
  
  @staticmethod
  def backward( ctx, grad_output ):
    s = grad_output.clone()
    if ctx._clip_grad is not None:
      # s[torch.abs(s) > ctx._clip_grad] = 0
      s = torch.clamp( s, min=-ctx._clip_grad, max=ctx._clip_grad )
    return s, None
    
class bernoulli_straight_through_sign(torch.autograd.Function):
  """ Samples from a bernoulli distribution in `forward()` and passes the
  gradient straight through in `backward()`.
  """

  @staticmethod
  def forward( ctx, p, slope=1 ):
    ctx._slope = slope
    b = torch.bernoulli( p )
    return b
  
  @staticmethod
  def backward( ctx, grad_output ):
    # Gradient is constant, only sensitive to sign
    s = torch.sign( grad_output )
    return ctx._slope * s, None # None because 'slope' is not a differentiable input
