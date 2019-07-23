""" Tools for creating a search space over neural net architectures.
"""

from collections import OrderedDict
import concurrent.futures
import enum
import functools
import itertools
import json
import logging
import math
import os
import os.path
import random
import types

import numpy

import nnsearch.backend.training
import nnsearch.feature as feature
import nnsearch.hcsearch as hcsearch
import nnsearch.mcts as mcts
import nnsearch.sample as sample
import nnsearch.worker

from nnsearch.nnspec import *

# ----------------------------------------------------------------------------

def zero( s ):
  return 0

@enum.unique
class Heuristic(enum.Enum):
  """ Available state evaluation heuristics. Members can be called directly,
  like `Heuristic.Zero( state )`.
  """
  Zero = functools.partial(zero)
  
  def __call__( self, *args ):
    return self.value( *args )
  
  @staticmethod
  def from_json( json_string ):
    for h in Heuristic:
      if h.name == json_string:
        return h

# ----------------------------------------------------------------------------

class InsertBeforeOutputSearchSpace:
  """ Inserts the next layer immediately before the output layer.
  """
  def __init__( self, initial_groups, actions, shuffle=False ):
    """
    Parameters:
      initial_groups : [LayerSpec]. List of "logical layers". New layers will
        be inserted just before the first layer in the last group.
    """
    self._initial_groups = SequenceSpec( *initial_groups )
    self._actions = actions
    self._shuffle = shuffle
    
  def start( self ):
    return self._initial_groups
    
  def actions( self, s ):
    return self._actions[:]

  def successors( self, s ):
    actions = self._actions
    if self._shuffle:
      actions = self._actions[:]
      random.shuffle( actions )
    for a in actions:
      cp = list(s)
      cp.insert( -1, a )
      yield SequenceSpec( *cp )
      
@enum.unique
class SearchStrategy(enum.Enum):
  """ Available search strategies.
  """
  InsertBeforeOutput  = InsertBeforeOutputSearchSpace

# ----------------------------------------------------------------------------

class AppendLayerAction:
  def __init__( self, layer ):
    self._layer = layer
    
  def layer( self ):
    return self._layer
    
  def terminal( self ):
    return self._layer.has_output_layer()
    
  def __call__( self, s ):
    # print( "Append to: " + str(s) )
    cp = list(s)
    cp.append( self._layer )
    # print( "\tlist: " + str(cp) )
    sprime = SequenceSpec( *cp )
    # print( "\tsprime: " + str(sprime) )
    return sprime
    
  def __str__( self ):
    return "Append({})".format( str(self._layer) )
    
  def __eq__( self, other ):
    return (type(self), self._layer) == (type(other), other._layer)
    
  def __hash__( self ):
    return hash( (type(self), self._layer) )
    
class OutputLayerAction(AppendLayerAction):
  def __init__( self, output_spec ):
    super().__init__( output_spec )

class TrainAndEvaluateAction:
  def __str__( self ):
    return "TrainAndEvaluate"
  
class InsertBeforeOutputAction:
  def __init__( self, layer ):
    self._layer = layer
    
  def __call__( self, s ):
    cp = list(s)                  # Unpack top-level SequenceSpec
    cp.insert( -1, self._layer )  # Insert layer
    return SequenceSpec( *cp )    # Re-pack
    
  def __str__( self ):
    return "InsertBeforeOutput({})".format( self._layer )
    
@enum.unique
class Actions(enum.Enum):
  InsertBeforeOutput = InsertBeforeOutputAction
  
  @staticmethod
  def from_json( json_dict, layer ):
    for a in Actions:
      if a.name == json_dict["name"]:
        return a.value( layer, **json_dict.get( "parameters", {} ) )
    
# ----------------------------------------------------------------------------

class NeuralNetworkSearchSpace:
  log_format = "[%(process)d]:%(levelname)s:%(name)s:%(message)s"
  
  def __init__( self, input, output, layers ):
    """
    Parameters:
      `base` : A search space over `LayerSpec`s
    """
    self._input = input
    self._output = output
    self._layers = layers

    log = logging.getLogger( __name__ )
    log.info( "Input: {}".format( input ) )
    # print( "Output: {}".format( output ) )
    log.info( "Output:" )
    for layer in self._output:
      log.info( str(layer) )
    log.info( "Layers:" )
    for layer in self._layers:
      log.info( str(layer) )
    
  def start( self ):
    return SequenceSpec( self._input )
    
  def terminal( self, s ):
    return isinstance(s[-1].flat()[-1], OutputLayerSpec)
    
  def actions( self, s ):
    # * FC layers can only be the final layers before output
    # * Maximum of 2 FC layers
    # * 2nd FC layer must be no larger than first
    n_fc_layers = sum( layer.has_fc_layer() for layer in s )
    if n_fc_layers >= 2:
      # Output will add 1 FC layer, reaching max (2)
      # Already have max number of FC layers -> can't append anything else
      # FIXME: See below
      assert( False )
      valid = [a for a in self._output]
    elif n_fc_layers > 0:
      # Have 1 FC layer -> can only append another FC layer
      valid = []
      # Size of last FC layer in last module
      fc_layers = s[-1].get_fc_layers()
      prev_n = fc_layers[-1].n()
      # FIXME: Workaround for ff: Output modules have to contain an FC layer,
      # so we only look in _output because this is the last FC layer allowed
      for layer in self._output:
        fc = layer.get_fc_layers()
        if len(fc) > 0 and fc[-1].n() <= prev_n:
          valid.append( layer )
    else:
      # FIXME: Wistuba2017 only allows FC layers after input dimension is
      # reduced to <= 8x8. We are using a *search space-specific* method of
      # detecting this: assume 2 pool layers is always enough to reduce
      # dimension to <= 8 (true in CIFAR10/MNIST with pool stride >= 2)
      n_pool_layers = sum( layer.has_pool_layer() for layer in s )
      if n_pool_layers >= 2:
        # Everything is allowed
        valid = self._output[:]
        valid.extend( self._layers[:] )
      else:
        # FC layers not allowed yet
        valid = [a for a in self._layers if not a.has_fc_layer()]
      # Pool only allowed after conv layer and if some conv layer > 1 size
      bigconv = any( any( conv.filter_size() > 1 for conv in layer.get_conv_layers() ) for layer in s )
      if not s[-1].has_conv_layer() or not bigconv:
        valid = [a for a in valid if not a.has_pool_layer()]
        
    actions = [AppendLayerAction(a) for a in valid]
    log = logging.getLogger( __name__ )
    log.debug( "Actions for {}".format( s.arch_string() ) )
    for a in actions:
      log.debug( str(a) )
    
    return actions
    
  def discount( self ):
    return 1.0

class SynchronousNasMdp:
  def __init__( self, model, pool ):
    self._model = model
    self._pool = pool
    
  def __getattr__( self, attr ):
    # Delegate anything not defined here to `model`
    return getattr( self._model, attr )
    
  def shutdown( self ):
    self._pool.shutdown()
    
  def sample_transition( self, rng, s, a ):
    sprime = a(s)
    if self.terminal( sprime ):
      f = self._pool.train_and_evaluate( sprime )
      r = f.result()
      return (sprime, r)
    else:
      return (sprime, 0)
    
# ----------------------------------------------------------------------------
    
class TrainRandomEvaluator:
  def __call__( self, rng, model, controller, path, sn ):
    logging.getLogger( __name__ ).info( "TrainRandomEvaluator(): %s", str(sn) )
    finished = False
    
    # Choose random action according to preference hierarchy
    ans = list(sn.successors())
    train_ans = [an for an in ans if an.action().layer().has_output_layer()]
    if len(train_ans) > 0:
      # Prefer actions that terminate the rollout
      an = rng.choice( train_ans )
      finished = True
    else:
      pool = [an for an in ans if an.action().layer().has_pool_layer()]
      if len(pool) > 0:
        # Prefer pooling actions, so that FC actions are allowed sooner
        an = rng.choice( pool )
      else:
        conv = [an for an in ans if an.action().layer().has_conv_layer()]
        bigconv = [an for an in conv if any( 
                    c.filter_size() > 1 for c in an.action().layer().get_conv_layers() )]
        if len(bigconv) > 0:
          # Prefer big convolutions, since they enable pooling actions
          an = rng.choice( bigconv )
        else:
          an = rng.choice( conv )
    
    # Similar to a recursive implementation of mcts.trajectory()
    sn.visit()
    an.visit()
    (sprime, r) = model.sample_transition( rng, sn.state(), an.action() )
    snprime = an.require_successor( sprime, controller )
    if not finished:
      # This results in a recursive call to self()
      q = snprime.expand( model, controller, path + [an, snprime] )
    else:
      q = 0
    q = r + model.discount() * q
    controller.update( sn, an, snprime, q )
    return q
  
    # -- v1
    # train_ans = [an for an in sn.successors() 
                 # if an.action().layer().has_output_layer()]
    # an = rng.choice( train_ans )
    # sn.visit()
    # an.visit()
    # (sprime, r) = model.sample_transition( rng, sn.state(), an.action() )
    # snprime = an.require_successor( sprime, controller )
    # controller.update( sn, an, snprime, r )
    # return r
    
@enum.unique
class Evaluator(enum.Enum):
  Constant = mcts.ConstantEvaluator
  Parent = mcts.ParentEvaluator
  TrainRandom = TrainRandomEvaluator
  
  @staticmethod
  def from_json( json_dict ):
    for e in Evaluator:
      if e.name == json_dict["name"]:
        return e.value( **json_dict["parameters"] )
    raise RuntimeError( "Invalid Evaluator '" + str(json_dict["name"]) + "'" )

class ControllerWrapper:
  def __init__( self, base, log_tree_interval=1 ):
    self._base = base
    self._ntrajectories = 0
    self._log_tree_interval = log_tree_interval
    
  def finish_trajectory( self, sn0 ):
    self._ntrajectories += 1
    if self._ntrajectories % self._log_tree_interval == 0:
      log = logging.getLogger( __name__ )
      log.info( "Search tree after %s trajectories", self._ntrajectories )
      nnsearch.mcts.print_tree( sn0, log.info )

  def __getattr__( self, attr ):
    # Delegate anything not defined here to `base`
    return getattr( self._base, attr )
    
@enum.unique
class Controller(enum.Enum):
  """ Available mcts controllers.
  """
  Uct = mcts.UctController
  MaxUct = mcts.MaxUctController
  
  @staticmethod
  def from_json( json_dict, model ):
    for c in Controller:
      if c.name == json_dict["name"]:
        parameters = dict( json_dict["parameters"] ) # Personal copy
        seed = parameters.pop( "seed", None )
        rng = random.Random( seed )
        interval = parameters.pop( "log_tree_interval", math.inf )
        evaluator = Evaluator.from_json( parameters.pop( "evaluator" ) )
        return ControllerWrapper( c.value( rng=rng, model=model, evaluator=evaluator, **parameters ),
                                  log_tree_interval=interval )
    raise RuntimeError( "Invalid controller '" + str(json_dict["name"]) + "'" )
    
class MctsAlgorithm:
  """ Encasulates the `mcts` algorithm with fixed parameters supplied as a
  JSON dictionary.
  """
  def __init__( self, json_dict ):
    self._parameters = json_dict

  def __call__( self, model ):
    controller = Controller.from_json( self._parameters["controller"], model )
    return mcts.mcts( model, controller )
    
@enum.unique
class SearchAlgorithm(enum.Enum):
  """ Available neural architecture search algorithms.
  """
  hcsearch = hcsearch.HcSearchAlgorithm
  mcts = MctsAlgorithm
  
  @staticmethod
  def from_json( json_dict ):
    for alg in SearchAlgorithm:
      if alg.name == json_dict["name"]:
        return alg.value( json_dict["parameters"] )
    raise RuntimeError( "Invalid search algorithm '" + str(json_dict["name"]) + "'" )

# ----------------------------------------------------------------------------

class WidthFeature(feature.Feature):
  def __init__( self ):
    super().__init__( "width" )
  def size( self ):
    return math.inf
  def __call__( self, s ):
    return s[-1].features()
    
class PoolFeature(feature.Feature):
  def __init__( self ):
    super().__init__( "pool" )
  def size( self ):
    return math.inf
  def __call__( self, s ):
    return sum( layer.has_pool_layer() for layer in s )
    
class ReceptiveFieldFeature(feature.Feature):
  def __init__( self ):
    super().__init__( "receptive_field" )
  def size( self ):
    return math.inf
  def __call__( self, s ):
    return s.receptive_field()
    
class LastConvFeature(feature.BooleanFeature):
  def __init__( self ):
    super().__init__( "last_conv" )
  def __call__( self, s ):
    return int(s[-1].has_conv_layer())
    
class LastFcFeature(feature.BooleanFeature):
  def __init__( self ):
    super().__init__( "last_fc" )
  def __call__( self, s ):
    return int(s[-1].has_fc_layer())
    
class HasBigConvFeature(feature.BooleanFeature):
  def __init__( self ):
    super().__init__( "has_bigconv" )
  def __call__( self, s ):
    return int(any( any(conv.filter_size() > 1 for conv in layer.get_conv_layers())
                    for layer in s ))
                
class ModuleLayersFeature(feature.Feature):
  def __init__( self ):
    super().__init__( "module_layers" )
  def size( self ):
    return math.inf
  def __call__( self, s ):
    return sum( 1 for _ in s[-1].learnable() )
    
class LayersFeature(feature.Feature):
  def __init__( self ):
    super().__init__( "layers" )
  def size( self ):
    return math.inf
  def __call__( self, s ):
    return sum( 1 for _ in s.learnable() )
    
class WistubaFeatureSpaceA(feature.FiniteFeatureSpace):
  def __init__( self ):
    # Note: Other parts of implementation depend on feature order
    features = [
      LastConvFeature(),
      LastFcFeature(),
      HasBigConvFeature(),
      feature.Enumerated(WidthFeature(), [128, 256, 512], default="min"),
      feature.Enumerated(PoolFeature(), [0, 1, 2], default="max"),
      feature.binned_contiguous(ReceptiveFieldFeature(), [1, 2, 5]),
      feature.binned_contiguous(LayersFeature(), [3, 7, 12])
    ]
    super().__init__( features )
    
class WistubaFeatureSpace_Test(feature.FiniteFeatureSpace):
  def __init__( self ):
    # Note: Other parts of implementation depend on feature order
    features = [
      LastConvFeature(),
      LastFcFeature(),
      HasBigConvFeature(),
      feature.Enumerated(WidthFeature(), [8, 16], default="min"),
      feature.Enumerated(PoolFeature(), [0, 1, 2], default="max"),
      feature.binned_contiguous(ReceptiveFieldFeature(), [1, 3]),
      feature.binned_contiguous(LayersFeature(), [3, 7, 12])
    ]
    super().__init__( features )

class CrossEntropyTabularModel:
  def __init__( self, rng, model, pool, feature_space, eval_epochs=1,
                mix_factor=0.5, shallowness=0.5, cumulative_training=False,
                gpu_logger = None ):
    self._rng = rng
    self._model = model
    self._pool = pool
    assert( 0 < mix_factor )
    assert( mix_factor <= 1 )
    self._mix_factor = mix_factor
    self._shallowness = shallowness
    self._eval_epochs = eval_epochs
    self._cumulative_training = cumulative_training
    self._gpu_logger = gpu_logger
    self._performance_cache = dict()
    
    self._feature_space = feature_space
    self._w = self._create_weight_dict( fill=True )
    
    log = logging.getLogger( __name__ )
    # Normalize new statistics and mix with old statistics
    log.info( "Initial policy:" )
    for fs in self._feature_space.states():
      # Don't update states we didn't encounter
      log.info( "  %s", fs )
      for a in self.actions( fs ):
        log.info( "    %s: %s", a, self._w[fs][a] )
    
  def _create_weight_dict( self, fill ):
    w = dict()
    for fs in self._feature_space.states():
      actions = self.actions( fs )
      if len(actions) == 0:
        w[fs] = None
      else:
        values = [0] * len(actions)
        if fill:
          # Give extra mass to actions that tend to make the network shorter
          ntotal = len(actions)
          special = [a.terminal() or a._layer.has_pool_layer() for a in actions]
          nspecial = sum(special)
          if nspecial > 0:
            # Everyone shares "ordinary" mass
            fill_value = (1.0 - self._shallowness) / ntotal
            # Special actions share bonus mass as well
            special_value = fill_value + self._shallowness / nspecial
          else:
            fill_value = 1.0 / ntotal
          for i in range(ntotal):
            if special[i]:
              values[i] = special_value
            else:
              values[i] = fill_value
        w[fs] = OrderedDict( zip(actions, values) )
    return w
  
  def actions( self, fs ):
    log = logging.getLogger( __name__ )
    # FIXME: Search will give wrong results if this function returns a different
    # set of actions from `model.actions(s)` (including different order!) for
    # any `s` that maps to `fs`. This is very fragile.
  
    # Note: This depends on order of self._features
    last_conv = 0
    last_fc = 1
    has_bigconv = 2
    width = 3
    pool = 4
    # * FC layers can only be the final layers before output
    # * Maximum of 2 FC layers
    # * 2nd FC layer must be no larger than first
    if fs[last_fc]:
      # Previous was FC layer -> can only append another FC layer
      valid = []
      # FIXME: Workaround for ff: Output modules have to contain an FC layer,
      # so we only look in _output because this is the last FC layer allowed
      actual_width = self._feature_space.feature(width).inverse(fs[width])
      for layer in self._model._output:
        fc = layer.get_fc_layers()
        if len(fc) > 0 and actual_width is not None and fc[-1].n() <= actual_width:
          valid.append( layer )
    else:
      # FIXME: Wistuba2017 only allows FC layers after input dimension is
      # reduced to <= 8x8. We are using a *search space-specific* method of
      # detecting this: assume 2 pool layers is always enough to reduce
      # dimension to <= 8 (true in CIFAR10/MNIST with pool stride >= 2)
      if fs[pool] >= 2:
        # Everything is allowed
        valid = self._model._output[:]
        valid.extend( self._model._layers[:] )
      else:
        # FC layers not allowed yet
        valid = [a for a in self._model._layers if not a.has_fc_layer()]
      # Pool only allowed after conv layer and if some conv layer > 1 size
      if not fs[last_conv] or not fs[has_bigconv]:
        valid = [a for a in valid if not a.has_pool_layer()]
    
    actions = [AppendLayerAction(a) for a in valid]
    log.debug( "Actions for {}".format( fs ) )
    for a in actions:
      log.debug( str(a) )
    return actions
      
  def _features( self, s ):
    return tuple( f(s) for f in self._features )
  
  # def _get_weights( self, s, fs, actions ):
    # w = self._w.get( fs, None )
    # if not w:
      # w = OrderedDict( (a, 1.0 / nactions) for a in actions )
      # self._w[fs] = w
    # return w
    
  def _sample_action( self, s ):
    # FIXME: This is fairly expensive to check, but it is necessary for now
    # because the two methods of generating actions can't share an implementation
    fs = self._feature_space.encode( s )
    actions = self._model.actions( s )
    feature_actions = self.actions( fs )
    if actions != feature_actions:
      log = logging.getLogger( __name__ )
      log.error( "Action sets not equal:" )
      log.error( "State: %s", str(s) )
      log.error( "actions: %s", list(map(str, actions)) )
      log.error( "Features: %s", str(fs) )
      log.error( "feature_actions: %s", list(map(str, feature_actions)) )
      assert( False )
    
    # w = self._get_weights( s, fs, actions )
    items = list( self._w[fs].items() )
    i = sample.multinomial( self._rng, [e[1] for e in items] )
    return items[i][0]
    
  def policy( self ):
    return self._w
  
  def propose( self ):
    s = self._model.start()
    path = [(None, s)]
    while not self._model.terminal( s ):
      a = self._sample_action( s )
      s = a(s)
      path.append( (a, s) )
    logging.getLogger( __name__ ).info( "ce.propose=%s", str(s) )
    return path
    
  def evaluate( self, samples ):
    log = logging.getLogger( __name__ )
    
    futures = []
    duplicates = dict()
    for (i, sample) in enumerate(samples):
      # `sample` is a list of `(a, s)` tuples
      s = sample[-1][1]
      # Detect duplicates
      dup = s in duplicates
      if not dup:
        duplicates[s] = i
      
      if dup:
        # Add another reference to the existing Future
        log.info( "Duplicate sample %s", str(s) )
        futures.append( futures[duplicates[s]] )
      elif not self._cumulative_training and s in self._performance_cache:
        # Use cached result if not doing cumulative training
        cached = self._performance_cache[s]
        log.info( "Using cached value for %s (%s)", str(s), str(cached) )
        f = concurrent.futures.Future()
        f.set_result( cached )
        futures.append( f )
      else:
        # Do work
        futures.append( self._pool.train_and_evaluate(
          s, nepoch=self._eval_epochs, gpu_logger=self._gpu_logger ) )
    # Wait for all evaluations
    concurrent.futures.wait( futures )
    # Select validation error
    values = [f.result()[0] for f in futures]
    for (sample, v) in zip(samples, values):
      s = sample[-1][1]
      self._performance_cache[s] = v
    logging.getLogger( __name__ ).info( "ce.evaluate=%s", values )
    return values
    
  def optimize( self, samples, weights ):
    log = logging.getLogger( __name__ )
    # Collect weights of sampled states and actions
    Nsa = self._create_weight_dict( fill=False )
    Ns = dict( (fs, 0) for fs in self._feature_space.states() )
    for (sample, w) in zip(samples, weights):
      if w == 0:
        continue
      for i in range(1, len(sample)):
        # `sample` is a list of `(a, s)` tuples
        s = sample[i-1][1]
        a = sample[i][0]
        sprime = sample[i][1]
        fs = self._feature_space.encode( s )
        Ns[fs]     += w
        Nsa[fs][a] += w
        
    # Normalize new statistics and mix with old statistics
    log.info( "Policy:" )
    for fs in self._feature_space.states():
      # Don't update states we didn't encounter
      if Ns[fs] == 0:
        continue
      log.info( "  %s", fs )
      Na = Nsa[fs]
      Z = sum(Na.values())
      for a in Na:
        Na[a] /= Z
        self._w[fs][a] *= (1.0 - self._mix_factor)
        self._w[fs][a] += self._mix_factor * Na[a]
        log.info( "    %s: %s", a, self._w[fs][a] )

# ----------------------------------------------------------------------------

class CrossEntropyModel:
  def __init__( self, rng, model ):
    self._rng = rng
    self._model = model
    self._w = dict() # Lazy initialization
    self._learning_rate = 0.1
    
    self._nfeatures = 7
      
  def _features( self, s ):
    f = []
    f.append( int(s[-1].has_conv_layer()) ) # Previous was conv layer?
    f.append( int(s[-1].has_fc_layer()) )   # Previous was fc layer?
    f.append( math.log( s[-1].features(), 2 ) ) # log_2 of current "width"
    f.append( sum( layer.has_pool_layer() for layer in s ) ) # Number of pool layers
    conv = [layer.get_conv_layers() for layer in s]
    f.append( sum( len(c) if c else 0 for c in conv ) ) # Number of conv layers
    fc = [layer.get_conv_layers() for layer in s]
    f.append( sum( len(c) if c else 0 for c in fc ) ) # Number of fc layers
    bigconv = any( any( conv.filter_size() > 1 for conv in layer.get_conv_layers() ) for layer in s )
    f.append( int(bigconv) ) # Any conv layers with filter_size > 1?
    assert( len(f) == self._nfeatures )
    return tuple(f)
  
  def _get_weights( self, s, f, nactions ):
    w = self._w.get( f, None )
    if not w:
      w = [[1.0] * self._nfeatures for _ in range(nactions)]
      self._w[f] = w
    return w
    
  def _sample_action( self, s ):
    actions = self._model.actions( s )
    f = self._features( s )
    w = self._get_weights( s, f, len(actions) )
    # Sample from Boltzmann distribution
    ps = [math.exp( sum( w[i][j] * f[j] for j in range(self._nfeatures) ) ) for i in range(len(w))]
    Z = sum(p)
    ps = [p / Z for p in ps]
    return sample.multinomial( self._rng, ps )
  
  def propose( self ):
    s = self._model.start()
    rewards = []
    while not model.terminal( s ):
      a = self._sample_action( s )
      (sprime, r) = self._model.sample_transition( self._rng, s, a )
      rewards.append( r )
    print( "propose " + str(s) )
    return (s, rewards)
    
  def evaluate( self, sr ):
    s, r = sr
    return sum(r)
    
  def optimize( self, candidates, weights ):
    sweights = [[0.01] * self._nactions for _ in range(self._nstates)]
    for (c, w) in zip(candidates, weights):
      # print( "candidate: " + str(w) + " : " + str(c) )
      h, r = c
      sprev = None
      for (a, s) in h:
        if sprev is not None:
          sweights[sprev][a] += w
        sprev = s
    for s in range(len(sweights)):
      Z = sum(sweights[s])
      sweights[s] = [w / Z for w in sweights[s]]
      for a in range(len(self._pi[s])):
        d = sweights[s][a] - self._pi[s][a]
        self._pi[s][a] += self._learning_rate * d
      Z = sum(self._pi[s])
      self._pi[s] = [p / Z for p in self._pi[s]]
    print( "Policy:" )
    for s in range(len(self._pi)):
      print( "s" + str(s) + ": " + str(self._pi[s]) )
  
# ----------------------------------------------------------------------------

# TODO: For testing only. Remove in production.
if __name__ == "__main__":
  layer_specs = [
    LayerSpec.from_arch_string( "conv-2-2-2-2" ),
    LayerSpec.from_arch_string( "act-lrelu-0.2" ),
    LayerSpec.from_arch_string( "pool-max-2-2-2-2" ),
    LayerSpec.from_arch_string( "fc-100" ),
    LayerSpec.from_arch_string( "drop-0.2" ),
    LayerSpec.from_arch_string( "bnorm" )
  ]
  for spec in layer_specs:
    print( spec.arch_string() )
  
  layer_sets = [
    SequenceSet( ConvolutionLayerSet( filters=[16, 32], filter_size=[1, 3], xstride=[1, 2], ystride=[1, 2] ),
                 ActivationLayerSet( fn=list(ActivationFunction), lrelu=[0.5, 0.9] ),
                 PoolLayerSet( fn=["max", "sum"], xpool=[2], ypool=[2], xstride=[2], ystride=[2] ) ),
    PoolLayerSet( fn=["max", "sum"], xpool=[4], ypool=[4], xstride=[4], ystride=[4] ),
    FullyConnectedLayerSet( n=[100, 200] ),
    DropoutLayerSet( p=[0.1, 0.2] ),
    BatchNormLayerSet()
  ]
  actions = list(itertools.chain.from_iterable( layer_sets ))
  for a in actions:
    print( a.arch_string() )
    
  model = InsertBeforeOutputSearchSpace( [InputLayerSpec( 1, 28, 28 ), OutputLayerSpec( 10 )], actions=actions )
  print( model.start() )
  
  # for sprime in model.successors( model.start() ):
    # print( sprime.arch_string() )
  
  # s1 = list(model.successors( model.start() ))
  # for s in random.sample( s1, 5 ):
    # for sprime in model.successors( s ):
      # print( sprime.arch_string() )
      
  with open( "test_spec.json" ) as fin:
    json_root = json.load( fin )
  search_space = parse_json_search_space_list( fin )[0]
  print( search_space.start().arch_string() )
  for s in search_space.successors( search_space.start() ):
    print( s.arch_string() )
  