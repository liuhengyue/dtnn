from   collections import defaultdict
import math

import epsilon.gymx as gymx
import epsilon.rl as rl
import epsilon.rl.policy as policy

# ----------------------------------------------------------------------------

def delta_L2( convergence_threshold = 1e-6 ):
  def f( dV ):
    sq = [d * d for d in dV.values()]
    norm = math.sqrt( sum(sq) )
    return norm < convergence_threshold
  return f
  
def maxV( mdp, V, s ):
  qstar = -math.inf
  astar = None
  for a in gymx.discrete_iter(mdp.action_space):
    qa = 0
    for (p, (sprime, r)) in mdp.P( s, a ):
      qa += p * (r + mdp.discount * V[sprime])
    if qa > qstar:
      qstar = qa
      astar = a
  return qstar, astar

def value_iteration( mdp, converged=delta_L2( 1e-6 ), max_iter=None ):
  """ The Value Iteration algorithm.
  
  Parameters:
    `mdp` : The MDP to solve
    `converged` : Callable taking a `dict` of value deltas for each updated
      state and returning Boolean indicating whether VI has converged.
    `max_iter` : Maximum number of iterations
  
  Returns:
    `dict` mapping states to optimal values
  """
  V = defaultdict(lambda: 0.0)
  i = 0
  while max_iter is None or i < max_iter:
    i += 1
    dV = defaultdict(lambda: 0.0)
    for s in gymx.discrete_iter(mdp.observation_space):
      if mdp.terminal( s ):
        continue
      vnew, _astar = maxV( mdp, V, s )
      dV[s] = vnew - V[s]
      V[s] = vnew
    if converged(dV):
      break
  return V

def q_from_v( mdp, V ):
  """ Converts a value function to a state-action value function ("Q-function").
  
  Parameters:
    `mdp` : The MDP
    `V` : The value function, a `dict` mapping states to values
    
  Returns:
    `dict`-of-`dict`s mapping states -> actions -> values
  """
  q = defaultdict(dict)
  for s in gymx.discrete_iter(mdp.observation_space):
    if mdp.terminal( s ):
      q[s] = {a: 0 for a in gymx.discrete_iter(mdp.action_space)}
      continue
    for a in gymx.discrete_iter(mdp.action_space):
      q[s][a] = sum( p * (r + mdp.discount * V[sprime])
                     for (p, (sprime, r)) in mdp.P( s, a ) )
  return q
  
def maxQ( Q, s ):
  qstar = -math.inf
  astar = None
  qs = Q[s]
  for a, q in qs.items():
    if q > qstar:
      qstar = q
      astar = a
  return qstar, astar
  
class TabularFunction:
  """ Wraps a function defined by a `dict` so that you can call it like
  `f(x)` rather than `f[x]`.
  """
  def __init__( self, d ):
    self._d = d
    
  def __call__( self, x ):
    return self._d[x]
  
class VGreedyPolicy(rl.ExplicitPolicy):
  def __init__( self, mdp, V ):
    self.mdp = mdp
    self.V = V
    
  def observe( self, s ):
    _qstar, astar = maxV( self.mdp, self.V, s )
    def sample_action( rng ):
      return astar
    return sample_action
  
  def distribution( self, s ):
    astar = self( None, s )
    yield (1.0, astar)
    
class QGreedyPolicy(rl.ExplicitPolicy):
  def __init__( self, Q ):
    self.Q = Q
    
  def observe( self, s ):
    _qstar, astar = maxQ( self.Q, s )
    def sample_action( rng ):
      return astar
    return sample_action
    
  def distribution( self, s ):
    astar = self( None, s )
    yield (1.0, astar)
    
# ----------------------------------------------------------------------------
  
if __name__ == "__main__":
  import gym
  env = gym.make( "FrozenLake-v0" ).unwrapped
  env.render()
  Vstar = value_iteration( env, discount=1.0 )
  print( "Vstar: {}".format( Vstar ) )
  Qstar = q_from_v( Vstar, env, discount=1.0 )
  print( "Qstar:" )
  for s in range(env.nS):
    print( Qstar[s] )
  pi = VGreedyPolicy( env, Vstar )
  for s in range(env.nS):
    print( "{} : {}".format( s, pi.action( None, s ) ) )
