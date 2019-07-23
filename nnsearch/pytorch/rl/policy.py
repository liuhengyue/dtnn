import logging

import numpy as np
import numpy.linalg

import gym

from   nnsearch.pytorch import rl
from   nnsearch.pytorch.rl import gymx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

class ConstantPolicy(rl.Policy):
  def __init__( self, action ):
    self.action = action
  
  def observe( self, s ):
    def action( rng ):
      return self.action
    return action

class UniformRandomPolicy(rl.Policy):
  def __init__( self, env ):
    if not isinstance(env.action_space, gym.spaces.Discrete):
      raise ValueError(
        "Don't know how to sample from {}".format(env.action_space) )
    self.action_space = env.action_space
  
  def observe( self, s ):
    def action( rng ):
      # TODO: Generalize to other spaces
      return rng.randrange(self.action_space.n)
    return action

class MixturePolicy(rl.Policy):
  def __init__( self, pi_less, pi_greater, epsilon ):
    """
    Parameters:
      `pi_less` : Execute with probability `epsilon`
      `pi_greater` : Execute with probability `1 - epsilon`
      `epsilon` : A `Hyperparameter` giving the schedule for epsilon.
    """
    super().__init__()
    self.pi_less = pi_less
    self.pi_greater = pi_greater
    self.epsilon = epsilon
    self._nsteps = 0
  
  def reset( self ):
    # Note that we don't reset the step count
    log.debug( "MixturePolicy.epsilon: %s", self.epsilon() )
    self.pi_less.reset()
    self.pi_greater.reset()
  
  def observe( self, s ):
    # FIXME: Should compute these lazily in action() and cache them
    pi_less_action = self.pi_less.observe( s )
    pi_greater_action = self.pi_greater.observe( s )
    self.epsilon.set_epoch( self._nsteps, nbatches=1 )
    epsilon = self.epsilon() # Capture current value
    self._nsteps += 1
    def action( rng ):
      if epsilon > 0 and rng.random() < epsilon:
        return pi_less_action( rng )
      else:
        return pi_greater_action( rng )
    return action
    
def epsilon_greedy( epsilon ):
  def make_policy( pi ):
    return EpsilonGreedyPolicy( pi, epsilon )
  return make_policy

# ----------------------------------------------------------------------------

def transition_matrix( mdp, policy ):
  assert( gymx.is_discrete( mdp.observation_space ) )
  assert( gymx.is_discrete( mdp.action_space ) )
  assert( isinstance(policy, rl.ExplicitPolicy) )
  states = sorted( gymx.discrete_iter(mdp.observation_space) )
  idx    = {s: i for (i, s) in enumerate(states)}
  N = len(states)
  # Initial state distribution
  s0 = np.zeros( N )
  for (p, s) in mdp.P_initial():
    s0[idx[s]] = p
  s0 /= np.sum( s0 )
  # Transition matrix under policy
  P = np.zeros( (N, N) )
  for (i, s) in enumerate(states):
    if mdp.terminal( s ): # Make the MDP recurrent
      P[i] = s0
      continue
    for (ap, a) in policy.distribution( s ):
      for (sp, (sprime, _r)) in mdp.P( s, a ):
        P[i, idx[sprime]] += ap * sp
    P[i] /= np.sum( P[i] )
  return P
  
def stationary_distribution( P, eps=1e-12 ):
  assert( len(P.shape) == 2 )
  assert( P.shape[0] == P.shape[1] )
  N = P.shape[0]
  # Solver wants the usual Ax = b form
  A = P.copy().transpose()
  A -= np.eye( N )
  A[-1,:] = 1
  b = np.zeros( N )
  b[-1] = 1
  sd = np.linalg.solve( A, b )
  sd[np.abs(sd) < eps] = 0
  return sd / np.sum(sd)
