import abc
import copy
import itertools
import logging

import numpy as np

import torch
import torch.nn.functional as fn
import torch.nn.utils as nnutils

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

class ReinforceEvaluationPolicy(rl.Policy):
  def __init__( self, policy_network ):
    super().__init__()
    self.policy_network = policy_network
    
  def observe( self, s ):
    def action( rng ):
      if not hasattr(action, "cached"):
        logits = self.policy_network( s )
        assert( len(logits.size()) == 2 and logits.size(0) == 1 )
        action.cached = torch.argmax( logits, dim=1 ).item()
      return action.cached
    return action
    
class ReinforceBanditLearner(rl.Learner):
  """ Base class for Q-learning algorithms. Derived classes must implement
  `target()` to compute the target for Q-learning, and can optionally override
  `_finish_batch()` to do things like updating a target network.
  """
  def __init__( self, env, policy_network, optimizer, temperature,
                batch_size, device=torch.device("cpu") ):
    super().__init__()
    self.env = env
    self.policy_network = policy_network
    self.optimizer   = optimizer
    self.temperature = temperature
    self.batch_size  = batch_size
    self.device = device
    self._current_batch = []
    self._train_step = 0
  
  def eval_policy( self ):
    return ReinforceEvaluationPolicy( self.policy_network )
    
  def _finish_batch( self ):
    pass
    
  def train( self, train=True ):
    self.policy_network.train( train )
    
  @property
  def training( self ):
    return self.policy_network.training
  
  def apply_to_modules( self, fn ):
    fn( "policy_network", self.policy_network )

  def training_episode( self, rng, episode, episode_length ):
    with torchx.training_mode( False, self ):
      v = 0
      o = self.env.reset()
      steps = (range(episode_length) if episode_length is not None
               else itertools.count())
      for t in steps:
        self.temperature.set_epoch( self._train_step, 1 )
        self._train_step += 1
      
        p = self.policy_network( o )
        temperature = self.temperature()
        log.verbose( "reinforce.train.temperature: %s", temperature )
        p /= temperature # Exploration
        p = torch.softmax( p, dim=1 )
        log.debug( "reinforce.train.p: %s", p )
        a = torch.multinomial( p, 1 ).item()
        log.debug( "reinforce.train.a: %s", a )
        oprime, r, done, info = self.env.step( a )
        # Don't want to store the replay memory on the GPU
        sample = (o.cpu(), a, oprime.cpu(), r.cpu(), done.cpu())
        v += r.item()
        with torchx.printoptions_nowrap( threshold=100000 ):
          log.micro( "t = %s: %s", t, sample )
        self._current_batch.append( sample )
        o = oprime
        
        if len(self._current_batch) >= self.batch_size:
          self._train_batch( self._current_batch )
          self._current_batch = []
          
        if done:
          break
      return v
    
  def _train_batch( self, batch ):
    log.verbose( "reinforce.train_batch" )
    for param_group in self.optimizer.param_groups:
      log.verbose( "reinforce.learning_rate: %s", param_group["lr"] )
    with torchx.training_mode( True, self ):
      # self.optimizer.zero_grad()
      self.policy_network.zero_grad()
    
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
      
      log.verbose( "reinforce.a:\n%s", a )
      logits = self.policy_network( x )
      log.verbose( "reinforce.logits:\n%s", logits )
      log.verbose( "reinforce.r:\n%s", r )
      logp = torch.log_softmax( logits, dim=1 )
      log.verbose( "reinforce.logp:\n%s", logp )
      pg = r.detach() * torch.gather(logp, 1, a)
      # pg = r.detach() * logp[a]
      log.verbose( "reinforce.pg:\n%s", pg )
      # Negate because we're minimizing
      pg_loss = -torch.sum( pg )
      
      pg_loss.backward()
      
      nnutils.clip_grad_norm( self.policy_network.parameters(), 10 )
      
      self.optimizer.step()
      self._finish_batch()

# ----------------------------------------------------------------------------
