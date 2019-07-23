import abc
import copy
import itertools
import logging

import numpy as np

import torch
import torch.nn.functional as fn

from   nnsearch.pytorch import rl
from   nnsearch.pytorch import torchx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

class DqnPolicy(rl.Policy):
  def __init__( self, dqn ):
    super().__init__()
    self.dqn = dqn
    
  def observe( self, s ):
    def action( rng ):
      if not hasattr(action, "cached"):
        q = self.dqn( s )
        assert( len(q.size()) == 2 and q.size(0) == 1 )
        action.cached = torch.argmax( q, dim=1 ).item()
      return action.cached
    return action
    
class EvaluationPolicy(DqnPolicy):
  def __init__( self, *args, **kwargs ):
    super().__init__( *args, **kwargs )
    
  # def reset( self ):
    # self.dqn.train( False )
    
# ----------------------------------------------------------------------------

class BatchMaker:
  def state_batch( self, x ):
    return torch.cat( x, dim=0 )
    
  def reward_batch( self, r ):
    return torch.cat( r, dim=0 )

class QLearnerBase(rl.Learner, metaclass=abc.ABCMeta):
  """ Base class for Q-learning algorithms. Derived classes must implement
  `target()` to compute the target for Q-learning, and can optionally override
  `_finish_batch()` to do things like updating a target network.
  """
  def __init__( self, env, discount, dqn, optimizer, explore,
                batch_size, replay, loss=fn.mse_loss, update_interval=1,
                device=torch.device("cpu") ):
    super().__init__()
    self.env = env
    self.discount    = discount
    self.dqn         = dqn
    self.optimizer   = optimizer
    self.explore     = explore
    self.batch_size  = batch_size
    self.replay      = replay
    self.loss        = loss
    self.update_interval = update_interval
    self.device = device
  
  def eval_policy( self ):
    return EvaluationPolicy( self.env, self.dqn )
  
  @abc.abstractmethod
  def target( self, r, xprime, terminal ):
    raise NotImplementedError
    
  def _finish_batch( self ):
    pass

  def training_episode( self, rng, episode, episode_length ):
    with torchx.training_mode( False, self ):
      v = 0
      o = self.env.reset()
      self.explore.reset()
      steps = (range(episode_length) if episode_length is not None
               else itertools.count())
      print("STEPS", steps)
      for t in steps:
        a = self.explore( rng, o )
        print(a)
        oprime, r, done, info = self.env.step( a )
        # Don't want to store the replay memory on the GPU
        sample = (o.cpu(), a, oprime.cpu(), r.cpu(), done.cpu())
        v += r.item()
        with torchx.printoptions_nowrap( threshold=100000 ):
          log.micro( "t = %s: %s", t, sample )
        self.replay.offer( sample )
        o = oprime
        
        if t % self.update_interval == 0:
          batch_ids = self.replay.sample( rng, self.batch_size )
          batch = self.replay.get( batch_ids )
          self._train_batch( batch )
          
        if done:
          break
      return v
    
  def _train_batch( self, batch ):
    log.verbose( "dqn.train_batch" )
    for param_group in self.optimizer.param_groups:
      log.verbose( "dqn.learning_rate: %s", param_group["lr"] )
  
    with torchx.training_mode( True, self ):
      self.optimizer.zero_grad()
    
      # (samples, idx, weights) = batch
      # list-of-tuples -> tuple-of-lists
      x, a, xprime, r, terminal = zip( *batch )
      log.micro( "dqn.batch:\n%s\n%s\n%s\n%s", x, a, xprime, r )
      x        = self.env.to_device( torchx.cat( x, dim=0 ) )
      xprime   = self.env.to_device( torchx.cat( xprime, dim=0 ) )
      r        = self.env.to_device( torch.cat( r, dim=0 ) )
      terminal = self.env.to_device( torch.cat( terminal, dim=0 ) )
      # TODO: `a` is the only thing assumed to be not-a-tensor. This is mainly
      # because `env` never returns an action, so there is nowhere in the API
      # to make a wrapper that converts `a`. Should we introduce our own
      # conversion method in the PyTorch gym wrappers?
      a = self.env.to_device( torch.tensor( [[ai] for ai in a] ) )
      
      q = self.dqn( x )
      q = torch.gather(q, 1, a)
      y = self.target( r, xprime, terminal )
      loss = self.loss( y, q )
      
      loss.backward()
      self.optimizer.step()
      self._finish_batch()

# ----------------------------------------------------------------------------

def soft_update(target, source, source_proportion):
  assert 0 <= source_proportion <= 1
  for tparam, sparam in zip(target.parameters(), source.parameters()):
    tparam.data.copy_(
      tparam.data*(1.0 - source_proportion) + sparam.data*source_proportion)

def hard_update(target, source):
  for tparam, sparam in zip(target.parameters(), source.parameters()):
    tparam.data.copy_(sparam.data)
    
class DqnLearner(QLearnerBase):
  """ Standard DQN learning rule with separate target network.
  """
  def __init__( self, *args, target_update_interval, **kwargs ):
    super().__init__( *args, **kwargs )
    self.target_update_interval = target_update_interval
    self._step_count = 0
    self.target_dqn = copy.deepcopy(self.dqn)
    
  def train( self, train=True ):
    self.dqn.train( train )
    self.target_dqn.train( train )
    
  @property
  def training( self ):
    assert self.dqn.training == self.target_dqn.training
    return self.dqn.training
  
  def eval_policy( self ):
    return EvaluationPolicy( self.target_dqn )
  
  def apply_to_modules( self, fn ):
    fn( "dqn", self.dqn )
    fn( "dqn_target", self.target_dqn )

  def target( self, r, xprime, terminal ):
    nonterminal = 1 - terminal.float()
    qprime = self.target_dqn( xprime ).detach()
    vprime, _ = torch.max( qprime, dim=1, keepdim=True )
    nonterminal = torchx.unsqueeze_right_as( nonterminal, vprime )
    nonterminal = nonterminal.type_as( vprime.data )
    y = r + self.discount * nonterminal * vprime
    return y
    
  def _finish_batch( self ):
    self._step_count += 1
    if self._step_count == self.target_update_interval:
      log.info( "Updating target network" )
      hard_update( self.target_dqn, self.dqn )
      self._step_count = 0

class DoubleDqnLearner(QLearnerBase):
  """ "Double-DQN" learning rule, where a* is computed from the agent network
  but its value is computed from the target network.
  """
  def __init__( self, *args, target_update_interval, **kwargs ):
    super().__init__( *args, **kwargs )
    self.target_update_interval = target_update_interval
    self._step_count = 0
    self.target_dqn = copy.deepcopy(self.dqn)
  
  def train( self, train=True ):
    self.dqn.train( train )
    self.target_dqn.train( train )
    
  @property
  def training( self ):
    assert self.dqn.training == self.target_dqn.training
    return self.dqn.training
  
  def eval_policy( self ):
    return EvaluationPolicy( self.target_dqn )
  
  def apply_to_modules( self, fn ):
    fn( "dqn", self.dqn )
    fn( "dqn_target", self.target_dqn )
  
  def target( self, r, xprime, terminal ):
    nonterminal = 1 - terminal.float()
    target_qprime = self.target_dqn( xprime ).detach()
    qprime = self.dqn( xprime ).detach()
    astar = torch.argmax( qprime, dim=1, keepdim=True )
    vprime = torch.gather( target_qprime, 1, astar )
    nonterminal = torchx.unsqueeze_right_as( nonterminal, vprime )
    nonterminal = nonterminal.type_as( vprime.data )
    y = r + self.discount * nonterminal * vprime
    return y
  
  def _finish_batch( self ):
    self._step_count += 1
    if self._step_count == self.target_update_interval:
      log.info( "Updating target network" )
      hard_update( self.target_dqn, self.dqn )
      self._step_count = 0
