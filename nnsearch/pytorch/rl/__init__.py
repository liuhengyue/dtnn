import torch
import torch.autograd

# {Tensor,Variable}.__bool__ was patched as below in PyTorch commit
# eaaf5486ba1f9d031f66af772ec033f82e667a0a
# to work with one-element tensors, but this didn't make it into Pytorch 0.3.1
def __new_Tensor_bool( self ):
  if self.numel() == 0:
    return False
  elif self.numel() == 1:
    return torch.squeeze(self)[0] != 0
  raise RuntimeError("bool value of " + torch.typename(self) +
                     " containing more than one object is ambiguous")

def __new_Variable_bool( self ):
  if self.data.numel() <= 1:
    return self.data.__bool__()
  raise RuntimeError("bool value of Variable containing " +
                     torch.typename(self.data) +
                     " with more than one object is ambiguous")

try:
  torch.ByteTensor.__bool__ = __new_Tensor_bool
  torch.FloatTensor.__bool__ = __new_Tensor_bool                     
  torch.LongTensor.__bool__ = __new_Tensor_bool
  torch.autograd.Variable.__bool__ = __new_Variable_bool
except TypeError:
  pass

try:
  from torch.cuda import FloatTensor
  torch.cuda.ByteTensor.__bool__ = __new_Tensor_bool
  torch.cuda.FloatTensor.__bool__ = __new_Tensor_bool                     
  torch.cuda.LongTensor.__bool__ = __new_Tensor_bool
except (ImportError, TypeError):
  pass
# ----------------------------------------------------------------------------

from nnsearch.pytorch.rl.core import (
  Learner, Policy, ExplicitPolicy,
  MarkovDecisionProcess, MdpEnvironment, DiscreteEnvironmentMdp,
  EpisodeObserver, EpisodeObserverList, EpisodeLogger, TrajectoryBuilder, 
  episode, transitions)
