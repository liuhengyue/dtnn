import abc
import itertools
import logging
import random

import gym
import gym.spaces as spaces

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

class Policy(metaclass=abc.ABCMeta):
  def reset( self ):
    pass

  @abc.abstractmethod
  def observe( self, s ):
    raise NotImplementedError()
  
  def __call__( self, rng, s ):
    return self.observe( s )( rng )
    
class ExplicitPolicy(Policy):
  @abc.abstractmethod
  def distribution( self, s ):
    pass
    
class Learner(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def eval_policy( self ):
    raise NotImplementedError()

  @abc.abstractmethod
  def training_episode( self, rng, episode_length ):
    raise NotImplementedError()
    
  @abc.abstractmethod
  def apply_to_modules( self, fn ):
    raise NotImplementedError()
    
# ----------------------------------------------------------------------------

class MarkovDecisionProcess(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def P_initial( self ):
    """ Returns an iterator over `(probability, state)` tuples.
    
    If possible, implementations should yield the elements in decreasing order
    of probability for best average-case complexity of `sample_initial()`.
    """
    pass

  @abc.abstractmethod
  def P( self, s, a ):
    """ Returns an iterator over `(probability, (state, reward))` tuples.
    
    If possible, implementations should yield the elements in decreasing order
    of probability for best average-case complexity of `sample_P()`.
    """
    pass
    
  @abc.abstractmethod
  def terminal( self, s ):
    """ Returns `True` if `s` is a terminal state.
    """
    pass
    
  @property
  @abc.abstractmethod
  def discount( self ):
    """ Discount rate.
    """
    pass
    
  def sample_initial( self, rng, n=1 ):
    """ Sample from `P_initial()`.
    
    This generic implementation iterates through `P_initial()`. Consider
    implementing custom sampling if `P_initial()` is large or constructing
    state objects is expensive.
    """
    return self._sample( rng, n, self.P_initial() )
    
  def sample_P( self, rng, s, a, n=1 ):
    """ Sample from `P()`.
    
    This generic implementation iterates through `P()`. Consider implementing
    custom sampling if `P()` is large or constructing state objects is
    expensive.
    """
    return self._sample( rng, n, self.P( s, a ) )
    
  def _sample( self, rng, n, gen ):
    # TODO: Might there be an advantage to return cumulative probabilities from
    # P()/P_initial() instead of pointwise probabilities? How often do we have
    # to do the transformation below?
    r = [rng.random() for _ in range(n)]
    F = 0.0
    choices = [None] * n
    for p, thing in gen:
      for i in range(n):
        if r[i] is None:
          continue
        # Unconditional assignment avoids issue when total prob < 1.0 due to
        # floating point imprecision.
        choices[i] = thing
        F += p
        if r[i] < F: # Found the i'th element
          r[i] = None
    return choices

# ----------------------------------------------------------------------------
    
class DiscreteEnvironmentMdp(MarkovDecisionProcess):
  """ Adapts a gym "discrete environment" (i.e. one that implements the
  `gym.envs.toy_text.discrete.DiscreteEnv` concept) to be an MDP.
  
  To implement `terminal(s)`, we change the state space to include an
  additional Boolean attribute. For specific domains, you may be able to avoid
  this extra attribute with a custom implementation.
  """
  def __init__( self, env, discount ):
    self.env = env
    self._discount = discount
    # Add attribute to state to indicate "terminal"
    self.observation_space = spaces.Tuple(
      (env.observation_space, spaces.Discrete(2)) )
    
  def P_initial( self ):
    for (i, p) in enumerate(self.env.isd):
      yield (p, i)
    
  def P( self, s, a ):
    o, done = s
    for (p, oprime, r, done) in self.env.P[o][a]:
      sprime = (oprime, done)
      yield (p, (sprime, r))
      
  def terminal( self, s ):
    _o, done = s
    return done
      
  @property
  def discount( self ):
    return self._discount
    
# ----------------------------------------------------------------------------

class MdpEnvironment(gym.Env, MarkovDecisionProcess):
  def __init__( self, mdp ):
    super().__init__()
    self.mdp = mdp
    self.rng = None
    self.s = None
    
    self.observation_space = self.mdp.observation_space
    self.action_space = self.mdp.action_space
    
  def close( self ):
    pass
    
  def render( self, mode="human" ):
    return self.mdp.render( self.s, mode )
    
  def seed( self, seed=None ):
    self.rng = random.Random( seed )
    
  @property
  def unwrapped( self ):
    return self
  
  def __str__( self ):
    return str(self.mdp)

  def reset( self ):
    self.done = False
    self.s = self.sample_initial( self.rng )[0]
    return self.s
    
  def step( self, a ):
    assert( not self.done )
    sprime, r = self.sample_P( self.rng, self.s, a )[0]
    done = self.terminal( sprime )
    self.s = sprime
    self.done = done
    return (self.s, r, self.done, {})
    
  def P_initial( self ):
    return self.mdp.P_initial()

  def P( self, s, a ):
    return self.mdp.P( s, a )
  
  def terminal( self, s ):
    return self.mdp.terminal( s )
    
  @property
  def discount( self ):
    return self.mdp.discount

# ----------------------------------------------------------------------------
    
class EpisodeObserver:
  def begin( self ):
    pass
    
  def end( self ):
    pass
    
  def observation( self, x ):
    pass
    
  def action( self, a ):
    pass
    
  def reward( self, r ):
    pass
    
  def info( self, info ):
    pass
    
class EpisodeObserverList(EpisodeObserver):
  def __init__( self, *observers ):
    self.observers = observers
    
  def begin( self ):
    for o in self.observers: o.begin()
    
  def end( self ):
    for o in self.observers: o.end()
    
  def observation( self, x ):
    for o in self.observers: o.observation( x )
  
  def action( self, a ):
    for o in self.observers: o.action( a )
  
  def reward( self, r ):
    for o in self.observers: o.reward( r )
  
  def info( self, info ):
    for o in self.observers: o.info( info )
    
class EpisodeLogger(EpisodeObserver):
  def __init__( self, log, level=logging.INFO, prefix="episode." ):
    self.log = log
    self.level = level
    self.prefix = prefix
    
  def begin( self ):
    self.log.log( self.level, "%sbegin", self.prefix )
    
  def end( self ):
    self.log.log( self.level, "%send", self.prefix )
    
  def observation( self, x ):
    self.log.log( self.level, "%sobservation: %s", self.prefix, x )
    
  def action( self, a ):
    self.log.log( self.level, "%saction: %s", self.prefix, a )
    
  def reward( self, r ):
    self.log.log( self.level, "%sreward: %s", self.prefix, r )
    
  def info( self, info ):
    self.log.log( self.level, "%sinfo: %s", self.prefix, info )
    
class TrajectoryBuilder(EpisodeObserver):
  def __init__( self ):
    self._traj = []
    
  @property
  def trajectory( self ):
    yield from self._traj
    
  @property
  def initial_state( self ):
    return self._traj[0]
    
  def begin( self ):
    self._traj = []
    
  def observation( self, x ):
    self._traj.append( x )
    
  def action( self, a ):
    self._traj.append( a )
    
  def reward( self, r ):
    self._traj.append( r )
    
# ----------------------------------------------------------------------------

def episode( rng, env, policy, observer=EpisodeObserver(), time_limit=None ):
  assert time_limit is None or time_limit >= 0
  v = 0
  x = env.reset()
  policy.reset()
  observer.begin()
  observer.observation( x )
  if time_limit == 0:
    observer.end()
    return (0, 0)
  steps = range(time_limit) if time_limit is not None else itertools.count()
  for t in steps:
    a = policy( rng, x )
    observer.action( a )
    x, r, done, info = env.step( a )
    observer.reward( r )
    observer.info( info )
    observer.observation( x )
    v += r
    if done:
      break
  observer.end()
  return (t+1, v)
  
def transitions( trajectory ):
  s = a = sprime = r = None
  # Need to peek one ahead to know whether sprime is a terminal state
  s = next(trajectory)
  for (i, e) in enumerate(trajectory):
    imod = ((i+1) % 3) # +1 Because we peeked
    if imod == 0: # State
      sprime = e
    elif imod == 1: # Action
      # We saw an action after sprime, thus sprime is not terminal
      if sprime is not None:
        yield (s, a, sprime, r, False)
        s = sprime
      a = e
    elif imod == 2:
      r = e
  yield (s, a, sprime, r, True)
