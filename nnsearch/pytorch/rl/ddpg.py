# Modified from original code: https://github.com/ghliu/pytorch-ddpg

import copy
import itertools
import logging

import numpy as np

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fn
from   torch.optim import Adam

import gym

import nnsearch.pytorch.torchx as torchx
import nnsearch.pytorch.rl as rl
import nnsearch.pytorch.rl.gymx as gymx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

class CriticModel(nn.Module):
  def __init__( self, head, tail ):
    """
    Parameters:
    -----------
      `head`: The portion of the critic that deals with the state only.
      `tail`: The portion that models both state and action. The input to
        `tail` is the output of `head` concatenated with the action.
    """
    super().__init__()
    self.head = head
    self.tail = tail
    
  def forward( self, s, a ):
    s = self.head( s )
    x = self.tail( torch.cat( [s, a], dim=1 ) )
    return x

def soft_update(target, source, source_proportion):
  assert 0 <= source_proportion <= 1
  for tparam, sparam in zip(target.parameters(), source.parameters()):
    tparam.data.copy_(
      tparam.data*(1.0 - source_proportion) + sparam.data*source_proportion)

def hard_update(target, source):
  for tparam, sparam in zip(target.parameters(), source.parameters()):
    tparam.data.copy_(sparam.data)
    
class ActorPolicy(rl.Policy):
  def __init__( self, actor ):
    self.actor = actor
    
  def observe( self, s ):
    def untorch( x ):
      if isinstance(x, tuple):
        return tuple( [untorch(xi) for xi in x] )
      # Actor network produces a batch of actions
      return x.data[0].cpu().numpy()
  
    a = self.actor( torchx.Variable(s) )
    def action( rng ):
      return untorch(a)
    return action
    
class RandomProcessExplorePolicy(rl.Policy):
  def __init__( self, env, base, explore_process ):
    self.env = env
    self.pi = base
    self.explore_process = explore_process
    
  def reset( self ):
    self.pi.reset()
    self.explore_process.reset()

  def observe( self, s ):
    # Noise is temporally correlated so we don't want to sample it every time
    delta = self.explore_process.sample()
    log.debug( "explore.delta: %s", delta )
    pi_s = self.pi.observe( s )
    def action( rng ):
      a = pi_s( rng ) + delta
      return np.clip( a, self.env.action_space.low, self.env.action_space.high )
    return action

class DDPGLearner(rl.Learner):
  def __init__( self, env, *, actor, critic, replay, discount, explore,
                actor_lr=1e-4, critic_lr=1e-3, batch_size=1,
                soft_update_mix=1e-3, update_interval=1 ):
    assert isinstance(env.action_space, gym.spaces.Box)
    action_shape = env.action_space.shape
    assert len(action_shape) == 1
    action_dim = action_shape[0]

    self.env = env
      
    self.actor        = actor
    self.actor_target = copy.deepcopy( actor )
    self.actor_optim  = Adam(self.actor.parameters(), lr=actor_lr)

    self.critic        = critic
    self.critic_target = copy.deepcopy( critic )
    self.critic_optim  = Adam(self.critic.parameters(), lr=critic_lr)
    
    self.replay = replay
    self.discount = discount
    self.explore = explore
    
    self.batch_size = batch_size
    self.soft_update_mix = soft_update_mix
    self.update_interval = update_interval
  
  def train( self, train=True ):
    self.actor.train( train )
    self.actor_target.train( train )
    self.critic.train( train )
    self.critic_target.train( train )
    
  @property
  def training( self ):
    assert self.actor.training == self.critic.training
    return self.actor.training
  
  def policy( self ):
    return ActorPolicy( self.actor_target )
  
  def training_episode( self, rng, episode_length=None ):
    with torchx.training_mode( False, self ):
      v = 0
      o = self.env.reset()
      self.explore.reset()
      steps = (range(episode_length) if episode_length is not None
               else itertools.count())
      for t in steps:
        a = self.explore( rng, o )
        log.debug( "ddpg.action: %s", a )
        oprime, r, done, info = self.env.step( a )
        v += r
        # Don't want to store the replay memory on the GPU
        sample = (o.cpu(), a, oprime.cpu(), r.cpu(), done.cpu())
        log.nano( "t = %s: %s", t, sample )
        self.replay.offer( sample )
        o = oprime
        
        if t % self.update_interval == 0:
          batch_ids = self.replay.sample( rng, self.batch_size )
          batch = self.replay.get( batch_ids )
          self._train_batch( batch )
          
        if done:
          break
      return v
    
  # @abc.abstractmethod
  def save_model( self, save_fn ):
    save_fn( "actor", self.actor )
    save_fn( "actor_target", self.actor_target )
    save_fn( "critic", self.critic )
    save_fn( "critic_target", self.critic_target )

  def _train_batch( self, batch ):
    with torchx.training_mode( True, self ):
      # list-of-tuples -> tuple-of-lists
      x, a, xprime, r, terminal = zip( *batch )
      if isinstance(self.env.observation_space, gym.spaces.Tuple):
        def cat( t ): return self.env.to_device( torch.cat( t, dim=0 ) )
        x = torchx.map_tuples( cat, x )
        xprime = torchx.map_tuples( cat, xprime )
      else:
        x = self.env.to_device( torch.cat( x, dim=0 ) )
        xprime = self.env.to_device( torch.cat( xprime, dim=0 ) )
      r        = self.env.to_device( torch.cat( r, dim=0 ) )
      terminal = self.env.to_device( torch.cat( terminal, dim=0 ) )
      # The convention for Gym environments with `Box` action space is that the
      # action is always a vector, even if it has only 1 element
      a = np.stack( a, axis=0 )
      a = self.env.to_device( torch.Tensor( a ) )
      log.nano( "ddpg.batch:\n%s\n%s\n%s\n%s\n%s", x, a, xprime, r, terminal )
      log.debug( "ddpg.rmean: %s", torch.mean(r) )
      
      x = torchx.Variable(x)
      a = torchx.Variable(a)
      xprime = torchx.Variable(xprime)
      r = torchx.Variable(r)
      terminal = torchx.Variable(terminal)
      
      # Critic update
      self.critic.zero_grad()
      # Predicted Q-values. Don't need gradients for this step
      # with torchx.no_grad(xprime):
      with torch.no_grad():
        qprime = self.critic_target( xprime, self.actor_target( xprime ) )
      qtarget = r + self.discount*(1.0 - terminal)*qprime
      
      q = self.critic( x, a )
      value_loss = fn.mse_loss( q, qtarget )
      log.debug( "ddpg.value_loss: %s", value_loss.item() )
      value_loss.backward()
      self.critic_optim.step()
      
      # Actor update
      self.actor.zero_grad()
      policy_loss = -self.critic( x, self.actor(x) )
      policy_loss = torch.mean( policy_loss )
      log.debug( "ddpg.policy_loss: %s", policy_loss.item() )
      policy_loss.backward()
      self.actor_optim.step()
      
      # Target update
      soft_update(self.actor_target, self.actor, self.soft_update_mix)
      soft_update(self.critic_target, self.critic, self.soft_update_mix)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
  import argparse
  import os
  import random
  from gym.envs import classic_control
  import nnsearch.logging as mylog
  import nnsearch.pytorch.modules as modules
  import nnsearch.pytorch.rl.gymx as gymx
  from nnsearch.pytorch.rl.random_process import OrnsteinUhlenbeckProcess
  from nnsearch.pytorch.rl.replay import CircularList
  
  parser = argparse.ArgumentParser()
  parser.add_argument( "--gpu", action="store_true" )
  parser.add_argument( "--batch", type=int, default=8 )
  parser.add_argument( "--replay-size", type=int, default=10000 )
  parser.add_argument( "--train-episodes", type=int, default=1000 )
  parser.add_argument( "--checkpoint-interval", type=int, default=100 )
  parser.add_argument( "--output", type=str, default="." )
  parser.add_argument( "--log-level", type=str, default="INFO" )
  args = parser.parse_args()
  
  # Logger setup
  # FIXME: Adding log levels should happen at package-level
  mylog.add_log_level( "MICRO",   logging.DEBUG - 5 )
  mylog.add_log_level( "NANO",    logging.DEBUG - 6 )
  mylog.add_log_level( "PICO",    logging.DEBUG - 7 )
  mylog.add_log_level( "FEMTO",   logging.DEBUG - 8 )
  mylog.add_log_level( "VERBOSE", logging.INFO - 5 )
  root_logger = logging.getLogger()
  root_logger.setLevel( mylog.log_level_from_string( args.log_level ) )
  # Need to set encoding or Windows will choke on ellipsis character in
  # PyTorch tensor formatting
  handler = logging.FileHandler( os.path.join(args.output, "ddpg.log"), "w", "utf-8")
  handler.setFormatter( logging.Formatter("%(levelname)s:%(name)s: %(message)s") )
  root_logger.addHandler(handler)

  rng = random.Random( 42 )
  to_device = (lambda x: x.cuda()) if args.gpu else None
  env = gymx.PyTorchEnvWrapper( classic_control.PendulumEnv(), to_device=to_device  )
  episode_length = 500
  
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  
  # Must convert to Python float or extreme weirdness ensues:
  # https://github.com/pytorch/pytorch/issues/4433
  action_range = (float(env.action_space.low[0]), float(env.action_space.high[0]))
  actor = nn.Sequential(
    nn.Linear( state_dim, 400, bias=False ), nn.ReLU(),
    nn.Linear( 400, 300, bias=False ), nn.ReLU(),
    nn.Linear( 300, action_dim ), nn.Tanh(),
    modules.Rescale( (-1, 1), action_range ) )
  critic = CriticModel(
    nn.Sequential( nn.Linear( state_dim, 400, bias=False ), nn.ReLU() ),
    nn.Sequential( nn.Linear( 400 + action_dim, 300, bias=False ), nn.ReLU(),
                   nn.Linear( 300, 1 ) ) )
  if args.gpu:
    actor = actor.cuda()
    critic = critic.cuda()
    
  replay = CircularList( args.replay_size )
  discount = 0.99
  explore = RandomProcessExplorePolicy( env, ActorPolicy( actor ),
    OrnsteinUhlenbeckProcess( theta=0.15, sigma=0.2 ) )
    
  learner = DDPGLearner( env, actor=actor, critic=critic, replay=replay,
    discount=discount, explore=explore, batch_size=args.batch )
  
  def evaluate( epoch, policy, episodes ):
    log.info( "Evaluate: epoch %s", epoch )
    T = 0
    V = 0
    for i in range(episodes):
      t, v = rl.episode( rng, env, policy, time_limit=episode_length,
                         observer=rl.EpisodeLogger( log, level=logging.MICRO ) )
      T += t
      V += v
    log.info( "eval.Vbar: %s", V / episodes )
    log.info( "eval.Tbar: %s", T / episodes )
  
  evaluate( 0, learner.policy(), 100 )
  for i in range(args.train_episodes):
    log.info( "Train: episode %s", i )
    learner.training_episode( rng, episode_length )
    if (i+1) % args.checkpoint_interval == 0:
      evaluate( i+1, learner.policy(), 100 )
