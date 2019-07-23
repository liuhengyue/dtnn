from   functools import reduce
import operator

import numpy as np

import gym
import gym.core
import gym.spaces

import torch
from   torch.autograd import Variable

def is_discrete( space ):
  return (isinstance(space, gym.spaces.Discrete)
          or (isinstance(space, gym.spaces.Tuple)
              and all(is_discrete(s) for s in space.spaces)))

def discrete_iter( space ):
  if isinstance(space, gym.spaces.Discrete):
    for i in range(space.n):
      yield i
  elif isinstance(space, gym.spaces.Tuple):
    def _component( spaces, i, partial=[] ):
      if i == len(spaces):
        yield tuple(partial)
      else:
        for j in range(spaces[i].n):
          yield from _component( spaces, i+1, partial + [j] )
    yield from _component(space.spaces, 0)
  else:
    raise TypeError( "unsupported space: {}".format(space) )

def discrete_size( space ):
  if isinstance(space, gym.spaces.Discrete):
    return space.n
  elif isinstance(space, gym.spaces.Tuple):
    return reduce(operator.mul, (discrete_size(sub) for sub in space.spaces))
  else:
    raise TypeError( "unsupported space: {}".format(space) )

def rescalar( x ):
  """ If `x` is a list with one element, return the element as a scalar. Else
  return `x` unchanged.
  """
  if x.size == 1:
    return x.item()
  else:
    return x

def unscalar( x ):
  """ If `x` is a scalar, wrap it in a list and return it. This assists in
  correctly constructing batched Tensors when the inputs might be scalars or
  Sequences.
  """
  return [x] if np.isscalar( x ) else x
  
def tensor_batch( x ):
  x = unscalar( x )
  t = torch.FloatTensor( x )
  t = t.unsqueeze(0)
  return t

class PyTorchEnvWrapper(gym.Wrapper):
  """ Adapts a `gym.Env` instance to return PyTorch `Tensor`s for observation,
  reward, and terminal. We do not convert actions because Gym does not
  represent them in a consistent way, and neural networks might represent them
  in idiosyncratic ways, too.
  
  All items are returned as a "batch" containing a single instance.
  """
  def __init__( self, env, to_device=None ):
    """
    Parameters:
    -----------
      `env` : Base gym.Env instance
      `to_device` : `Tensor -> Tensor` Put a tensor on the desired device.
        Default: identity function.
    """
    super().__init__( env )
    self.to_device = to_device if to_device is not None else lambda t: t
    
  def _tensor( self, x ):
    x = unscalar( x )
    t = torch.FloatTensor( x )
    t = t.unsqueeze(0)
    return self.to_device(t)
    # return Variable(self.to_device(t))

  def reset( self, **kwargs ):
    o = self.env.reset( **kwargs )
    # print( "reset.o: shape = {}".format( o.shape ) )
    return self.observation( o )
  
  def step( self, action ):
    assert action.shape == self.env.action_space.shape
    o, r, done, info = self.env.step( action )
    # print( "step.o: shape = {}".format( o.shape ) )
    return self.observation(o), self.reward(r), self.terminal(done), info

  def observation( self, o ):
    return self._tensor( o )
    
  def reward( self, r ):
    return self._tensor( r )
    
  def terminal( self, done ):
    return self._tensor( done )
