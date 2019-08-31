import abc
import itertools
import logging
import math
import numpy as np

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fn

import nnsearch.pytorch.gated.act as act
import nnsearch.pytorch.gated.relax as relax
import nnsearch.pytorch.torchx as torchx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------
# Count-based gating

class ConstantCount(nn.Module):
  def __init__( self, n, c ):
    super().__init__()
    self._n = n
    self._c = c
  
  def forward( self, x ):
    batch_size = x.size(0)
    nactive = Variable(self._c * torch.ones(batch_size).type_as(x.data))
    return nactive
    
class UniformCount(nn.Module):
  def __init__( self, n ):
    super().__init__()
    self._n = n

  def forward( self, x ):
    batch_size = x.size(0)
    p = Variable(torch.ones(batch_size, self._n+1).type_as(x.data))
    nactive = torch.multinomial( p, 1 ).squeeze().float()
    # log.micro( "UniformCount.nactive: %s", nactive )
    return nactive
    
class BinomialCount(nn.Module):
  def __init__( self, n, p ):
    super().__init__()
    self._n = n
    self._p = p

  def forward( self, x ):
    batch_size = x.size(0)
    p = Variable(self._p * torch.ones(batch_size, self._n+1).type_as(x.data))
    g = torch.bernoulli( 1 - p )
    log.micro( "BinomialCount.g: %s", g )
    nactive = torch.sum( g, dim=1 )
    log.micro( "BinomialCount.nactive: %s", nactive )
    return nactive
    
class PlusOneCount(nn.Module):
  def __init__( self, base ):
    super().__init__()
    self._base = base
    
  def forward( self, x ):
    nactive = self._base( x )
    g = 1 + nactive
    # log.micro( "PlusOneCount.nactive: %s", g )
    return g

class CountGate(nn.Module, metaclass=abc.ABCMeta):
  def __init__( self, ncomponents, count, gate_during_eval=False ):
    """
    Let C_n be the set of active components, where n = |C_n|. For the
    `NestedGate` strategy, C_{n-1} \subset C_n.
    
    Parameters:
      `count`: `x` -> integer in {0, ..., ncomponents} [batch_size]
      `gate_during_eval`: Enable dropout during evaluation phase
    """
    super().__init__()
    self.ncomponents = ncomponents
    self.count = count
    self.gate_during_eval = gate_during_eval
  
  @abc.abstractmethod
  def _arrange_gate_matrix( self, g ):
    return NotImplemented
    
  def forward( self, x ):
    batch_size = x.size(0)
    n = self.ncomponents
    c = self.count( x )
    if not self.training and not self.gate_during_eval:
      g = Variable(torch.ones( batch_size, n ).type_as(x.data))
      # print("eval mode: ", g)
    else:
      log.debug( "c=%s", c )
      g = relax.gate_matrix_from_count.apply( c, n )
      g = self._arrange_gate_matrix( g )
      log.debug( "g=%s", g )
      # print("train/test mode: ", g)
    # g is probably on different device from x, make sure they are on same one
    g = g.to(x.device)
    return g
    
class NestedCountGate(CountGate):
  """
  Let C_n be the set of active components, where n = |C_n|. For the
  `NestedGate` strategy, C_{n-1} \subset C_n.
  
  Parameters:
    `nactive_fn`: `batch_size` -> integer in {0, ..., ncomponents} [batch_size]
    `normalize`: Divide the 0-1 gate matrix by the number of active components
    `gate_during_eval`: Enable dropout during evaluation phase
  """
  def __init__( self, *args, **kwargs ):
    super().__init__( *args, **kwargs )
  # When training gatedchainnetwork, an error will occur:
  # 'NestedCountGate' has no attribute 'set_control'
  # it does not need this, since it will activate the first c components
  # given the number of count, so u is not used
  # def set_control( self, u ):
  #   self._u = u
  def _arrange_gate_matrix( self, g ):
    return g


class NestedCountFromUGate(CountGate):
    """
    Let C_n be the set of active components, where n = |C_n|. For the
    `NestedGate` strategy, C_{n-1} \subset C_n.

    Parameters:
      `nactive_fn`: `batch_size` -> integer in {0, ..., ncomponents} [batch_size]
      `normalize`: Divide the 0-1 gate matrix by the number of active components
      `gate_during_eval`: Enable dropout during evaluation phase
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _arrange_gate_matrix(self, g):
        return g

    def set_control(self, u):
        self._u = u

    def forward( self, x ):
        batch_size = x.size(0)
        n = self.ncomponents
        c = self.count( self._u)
        if not self.training and not self.gate_during_eval:
            g = Variable(torch.ones( batch_size, n ).type_as(x.data))
          # print("eval mode: ", g)
        else:
            log.debug( "c=%s", c )
            g = relax.gate_matrix_from_count.apply( c, n )
            g = self._arrange_gate_matrix( g )
            log.debug( "g=%s", g )
          # print("train/test mode: ", g)
        # g is probably on different device from x, make sure they are on same one
        g = g.to(x.device)
        return g

class RandomPermutationCountGate(CountGate):
  def __init__( self, *args, **kwargs ):
    super().__init__( *args, **kwargs )

  def _arrange_gate_matrix( self, g ):
    # Different permutation for each row
    perms = [torch.randperm(g.size(1)) for _ in range(g.size(0))]
    p = torch.stack( perms, dim=0 ).type_as(g.data).long()
    return torch.gather(g, 1, p)
    
# ----------------------------------------------------------------------------
# Component-based gating

class BernoulliGate(nn.Module):
  def __init__( self, clip_grad=None ):
    super().__init__()
    self._clip_grad = clip_grad

  def forward( self, x ):
    g = relax.bernoulli_straight_through.apply( x, self._clip_grad )
    log.debug( "g=%s", g )
    return g
    
class BernoulliSignGate(nn.Module):
  def forward( self, x ):
    log.debug( "x=%s", x )
    g = relax.bernoulli_straight_through_sign.apply( x )
    log.debug( "g=%s", g )
    return g
    
class ThresholdGate(nn.Module):
  def __init__( self, clip_grad=None ):
    super().__init__()
    self._clip_grad = clip_grad
    
  def forward( self, x ):
    g = relax.threshold_straight_through.apply( x, self._clip_grad )
    log.debug( "g=%s", g )
    return g
    
class ThresholdSignGate(nn.Module):
  def forward( self, x ):
    g = relax.threshold_straight_through_sign.apply( x )
    log.debug( "g=%s", g )
    return g
    
class ActOneShotGate(nn.Module):
  def __init__( self, epsilon=0.01 ):
    super().__init__()
    self.epsilon = epsilon

  def forward( self, x ):
    g, rho = act.act_one_shot( x, self.epsilon )
    log.debug( "g=%s", g )
    return (g, rho)
    
class ReinforceGate(nn.Module):
  def forward( self, x ):
    g = torch.bernoulli( x )
    log.debug( "g=%s", g )
    # REINFORCE needs the probabilities
    return (g, x)
    
class ReinforceGateHard(nn.Module):
  def forward( self, x ):
    g = x.clone()
    g[g >= 0.5] = 1
    g[g <  0.5] = 0
    log.debug( "g=%s", g )
    # REINFORCE needs the probabilities
    return (g, x)
    
class ReinforceCountGate(nn.Module):
  def forward( self, x ):
    c = torch.multinomial( x, 1 ).squeeze().type_as(x.data)
    n = x.size(1) - 1 # 'c' ranges from {0,...,n} (= n+1 choices)
    g = relax.gate_matrix_from_count.apply( c, n )
    log.debug( "g=%s", g )
    # REINFORCE needs the probabilities
    return (g, x)
    
class ConcreteBernoulliGate(nn.Module):
  def __init__( self, temperature ):
    super().__init__()
    self._temperature = temperature
    
  @property
  def temperature( self ):
    return self._temperature
    
  @temperature.setter
  def temperature( self, t ):
    assert( t >= 0 )
    self._temperature = t
  
  def forward( self, p ):
    # Params are in [0,1] -> Need them in (0,1) so log is defined
    # FIXME: Would rather determine type from `p`, but `p[0]` always returns
    # `float` even when it's actually a `np.float32`.
    eps = float( np.finfo(np.float32).eps )
    psafe = (p + eps) / (1.0 + 2*eps)
    logp = torch.log( psafe ) - torch.log( 1.0 - psafe )
    u = Variable( torch.zeros_like( p.data ).uniform_() )
    usafe = (u + eps) / (1.0 + 2*eps)
    logu = torch.log( usafe ) - torch.log( 1.0 - usafe )
    testval = logp + logu
    # Deterministic limit
    if self.temperature == 0:
      return (testval >= 0).type_as(p)
    else:
      return fn.sigmoid( testval / self.temperature )
      
class ConcreteCountGate(nn.Module):
  def __init__( self, temperature ):
    super().__init__()
    self._temperature = temperature
    
  @property
  def temperature( self ):
    return self._temperature
    
  @temperature.setter
  def temperature( self, t ):
    assert( t >= 0 )
    self._temperature = t
  
  def forward( self, alpha ):
    # Params are in [0,1] -> Need them in (0,1) so log is defined
    # FIXME: Would rather determine type from `p`, but `p[0]` always returns
    # `float` even when it's actually a `np.float32`.
    batch_size = alpha.size(0)
    ncomps = alpha.size(1)
    eps = float( np.finfo(np.float32).eps )
    alpha_safe = alpha + eps
    log_alpha = torch.log( alpha_safe )
    U = Variable( torch.zeros_like( log_alpha.data ).uniform_() )
    Usafe = (U + eps) / (1.0 + 2*eps)
    G = -torch.log( -torch.log( Usafe ) )
    testval = log_alpha + G
    # Deterministic limit
    if self.temperature == 0:
      _, c = torch.max( testval, dim=1 )
      return relax.gate_matrix_from_count( c, ncomps )
    else:
      # BxC matrix of count weights
      w = fn.softmax( testval / self.temperature )
      # All C BxC nested count matrices with rows [1 1 ... 1 0 0 ... 0]
      gmix = [relax.gate_matrix_from_count(c.expand(batch_size, -1), ncomps)
              for c in range(ncomps)]
      # BxCxC stack of all count matrices
      gmix = torch.stack( gmix, dim=2 )
      # Every count matrix weighted by its component's weight
      g = w.expand(batch_size, ncomps, ncomps) * gmix
      # Convex combination of weighted count matrices
      g = torch.sum( g, dim=2 )
      return g

# ----------------------------------------------------------------------------

class ProportionToCount(nn.Module):
  def __init__( self, minval, maxval ):
    super().__init__()
    self._minval = minval
    self._maxval = maxval
    
  def forward( self, u ):
    d = self._maxval - self._minval + 1
    log.debug( "ProportionToCount: range: [%s, %s] (%d)",
               self._minval, self._maxval, d )
    max_tensor = torch.tensor( [self._maxval], device=u.device ).type_as(u.data)

    c = torch.min( self._minval + torch.floor(u * d), max_tensor)
    log.debug( "ProportionToCount.c: %s", c )
    # c = torch.min( self._minval + torch.floor(u * d), self._maxval )
    return c

class CountToNestedGate(nn.Module):
  def __init__( self, ncomponents ):
    super().__init__()
    self.ncomponents = ncomponents

  def forward( self, c ):
    return relax.gate_matrix_from_count.apply( c, self.ncomponents )
    
class PermuteColumns(nn.Module):
  def forward( self, g ):
    # Different permutation for each row
    perms = [torch.randperm(g.size(1)) for _ in range(g.size(0))]
    p = torch.stack( perms, dim=0 )
    if g.data.is_cuda:
      p = p.cuda()
    p = Variable(p)
    return torch.gather(g, 1, p)

class TemperedSigmoidGate(nn.Module):
  def __init__( self, temperature ):
    super().__init__()
    self._temperature = temperature
    
  @property
  def temperature( self ):
    return self._temperature
    
  @temperature.setter
  def temperature( self, t ):
    assert( t >= 0 )
    self._temperature = t
    
  def forward( self, p ):
    if self.temperature == 0:
      return (p >= 0).type_as(p)
    else:
      exp = torch.exp( -p / self.temperature )
      return 1.0 / (1.0 + exp)
      
# ----------------------------------------------------------------------------

class GateController(metaclass=abc.ABCMeta):
  # FIXME: Add these
  # @property
  # @abc.abstractmethod
  # def gate_during_eval( self ):
    # raise NotImplementedError()
  
  # @gate_during_eval.setter
  # @abc.abstractmethod
  # def gate_during_eval( self, b ):
    # raise NotImplementedError()

  @abc.abstractmethod
  def reset( self, x ):
    raise NotImplementedError()
    
  @abc.abstractmethod
  def next_module( self, m ):
    raise NotImplementedError()
    
class GatePolicy(GateController):
  @abc.abstractmethod
  def set_control( self, u ):
    raise NotImplementedError()
    
class StaticGate(nn.Module, GatePolicy):
  def __init__( self, gate ):
    super().__init__()
    self.base = gate

  def reset( self, x ):
    pass
    
  def next_module( self, m ):
    raise NotImplementedError()
    
  def set_control( self, u ):
    self._u = u
    
  def forward( self, x ):
    return self.base( self._u )

class SequentialGate(nn.Module, GatePolicy):
  """ Executes a list of gate modules in order.
  """
  def __init__( self, gate_modules ):
    super().__init__()
    self.gate_modules = nn.ModuleList( gate_modules )
    
  def reset( self, x ):
    self._i = -1
    
  def next_module( self, m ):
    self._i += 1
    
  def set_control( self, u ):
    for m in self.gate_modules:
      m.set_control( u )
    
  def forward( self, x ):
    g = self.gate_modules[self._i]
    return g( x )
    
class JointGate(nn.Module, GatePolicy):
  """ Computes a single joint matrix on the first invocation, and returns
  slices of it for each gated module.
  """
  def __init__( self, gate, ncomponents ):
    """
    Parameters:
    -----------
      `gate`: Gate network
      `ncomponents`: List of number of components in each GatedModule
    """
    super().__init__()
    self.base = gate
    self.ncomponents = ncomponents
    self.boundaries = [0] + list(itertools.accumulate(ncomponents))
    
  def reset( self, x ):
    def expand( gout ):
      if isinstance(gout, tuple):
        g, info = gout # Fail fast on unexpected extra outputs
        return (g, info)
      else:
        return (gout, None)
  
    self.base.reset( x )
    self._gcache = expand( self.base( x ) )
    self._i = -1
    
  def next_module( self, m ):
    self._i += 1
  
  def set_control( self, u ):
    self.base.set_control( u )
    
  def forward( self, x ):
    i, j = self.boundaries[self._i], self.boundaries[self._i + 1]
    # log.debug( "joint.gcache:\n%s", self._gcache )
    ys = [(y[:,i:j].contiguous() if y is not None else None)
          for y in self._gcache]
    return tuple(ys)
    # gi = self._gcache[:,i:j].contiguous()
    # return gi
    
class BlockdropNestedGate(nn.Module, GatePolicy):
  """ Computes a single joint matrix on the first invocation, and returns
  slices of it for each gated module.
  """
  def __init__( self, ncomponents ):
    """
    Parameters:
    -----------
      `gate`: Gate network
      `ncomponents`: List of number of components in each GatedModule
    """
    super().__init__()
    self.count = StaticGate( ProportionToCount( 0, sum(ncomponents) ) )
    self.ncomponents = ncomponents
    self.boundaries = [0] + list(itertools.accumulate(ncomponents))
    
  def reset( self, x ):
    def expand( gout ):
      if isinstance(gout, tuple):
        g, info = gout # Fail fast on unexpected extra outputs
        return (g, info)
      else:
        return (gout, None)
        
    def allocate( c, i ):
      n = c[i].item()
      gi = [0] * len(self.ncomponents)
      s = 0
      while True:
        for j in reversed(range(len(gi))):
          if s == n:
            return gi
          if gi[j] < self.ncomponents[j] * self.count._u[i]:
            gi[j] += 1
            s += 1
  
    self.count.reset( x )
    c = self.count( x )
    log.debug( "blockdrop_nested.c: %s", c )
    
    cs = []
    for i in range(c.size(0)):
      cs.append( allocate( c, i ) )
    cs = torch.tensor( cs )
    log.debug( "blockdrop_nested.cs: %s", cs )
    
    gs = [relax.gate_matrix_from_count.apply( c, n ).type_as(self.count._u).float()
          for (c, n) in zip([cs[:,i] for i in range(cs.size(1))], self.ncomponents)]
    gs = torch.cat( gs, dim=1 )
    log.debug( "blockdrop_nested.gs: %s", gs )
    
    self._gcache = expand( gs )
    self._i = -1
    
  def next_module( self, m ):
    self._i += 1
  
  def set_control( self, u ):
    self.count.set_control( u )
    
  def forward( self, x ):
    i, j = self.boundaries[self._i], self.boundaries[self._i + 1]
    ys = [(y[:,i:j].contiguous() if y is not None else None)
          for y in self._gcache]
    return tuple(ys)
    # gi = self._gcache[:,i:j].contiguous()
    # return gi
    
# ----------------------------------------------------------------------------
    
class GateChunker(nn.Module, GatePolicy):
  """ Adapts an arbitrary gate strategy to output the same gate matrix for all
  the modules in contiguous "chunks" of specified sizes. Less flexible than
  `GroupedGate`, which can have non-contiguous groups, but works with things
  like RNN-based gate controllers.
  """
  def __init__( self, gate, chunk_sizes ):
    super().__init__()
    self.base = gate
    self.chunk_sizes = chunk_sizes
    
  def reset( self, x ):
    self.base.reset( x )
    self._gcache = None
    self._count = 0
    self._chunk = 0
    
  def next_module( self, m ):
    if self._count == self.chunk_sizes[self._chunk]:
      self._gcache = None
      self._count = 0
      self._chunk += 1
    if self._gcache is None:
      self.base.next_module( m )
    self._count += 1
  
  def set_control( self, u ):
    self.base.set_control( u )
    
  def forward( self, x ):
    if self._gcache is None:
      g = self.base( x )
      self._gcache = g
    else:
      g = self._gcache
    return g
    
class GroupedGate(nn.Module, GatePolicy):
  """ Computes each gate module once and uses the result for all modules in
  the corresponding group.
  """
  def __init__( self, gate_modules, groups ):
    """
    Parameters:
    `gate_modules` : List of gate modules, where 
      `len(gate_modules) == max(groups) - 1`
    `groups` : List of integer indices in {0, ..., `len(gate_modules) - 1`}
    """
    super().__init__()
    self.gate_modules = nn.ModuleList( gate_modules )
    self.groups = groups
    self.reset()
    
  def reset( self ):
    self._i = -1
    self._gcache = [None for _ in range(len(self.groups))]
  
  def next_module( self, m ):
    self._i += 1
    
  def set_control( self, u ):
    for m in self.gate_modules:
      m.set_control( u )
    
  def forward( self, x ):
    group = self.groups[self._i]
    g = self._gcache[group]
    if g is None:
      m = self.gate_modules[group]
      g = m( x )
      self._gcache[group] = g
    return g
    
# ----------------------------------------------------------------------------

if __name__ == "__main__":
  testval = Variable(torch.Tensor( [[0.1, 0.3, 0.6], [0.6, 0.1, 0.3]] ))
  temperature = 1
  batch_size = testval.size(0)
  ncomps = testval.size(1) - 1

  # BxC matrix of count weights
  w = fn.softmax( testval / temperature, dim=1 )
  print( w )
  # All C BxC nested count matrices with rows [1 1 ... 1 0 0 ... 0]
  gmix = [relax.gate_matrix_from_count.apply(c*torch.ones(batch_size), ncomps)
          for c in range(ncomps + 1)]
  # BxCxC stack of all count matrices
  gmix = torch.stack( gmix, dim=2 )
  print( gmix )
  # Every count matrix weighted by its component's weight
  wex = w.unsqueeze(1).expand(batch_size, ncomps, ncomps+1)
  print( wex )
  g = wex * gmix
  print( g )
  # Convex combination of weighted count matrices
  g = torch.sum( g, dim=2 )
  print( g )
