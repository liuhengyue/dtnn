import abc
import math

class Feature(abc.ABC):
  def __init__( self, name ):
    self._name = name
    
  def name( self ):
    return self._name

  @abc.abstractmethod
  def size( self ):
    return 0
  
  @abc.abstractmethod
  def __call__( self, s ):
    return None
    
class BooleanFeature(Feature):
  def __init__( self, name ):
    super().__init__( name )
    
  def size( self ):
    return 2

class Enumerated(Feature):
  def __init__( self, feature, values, default=None, name=None ):
    """ Maps a feature with a finite set of values to the set
      `{0, 1, ..., len(values)}`. Values that do not appear in `values` either
      raise an exception or are mapped to a special value, depending on
      `default`.
      
      Parameters:
        `feature` : Base feature
        `values` : List of values that should have distinct indices
        `default` : `{None, "min", "max"}` If not `None`, map non-enumerated
          values to a special value that is either the min or max mapping.
    """
    if default is not None and default != "min" and default != "max":
      raise ValueError( "Invalid argument: default ({}) \{\"min\", \"max\"\}".format( default ) )
    
    self._name = "Enumerated({})".format(feature.name()) if not name else name
    self._feature = feature
    self._default = default
    d = 1 if self._default == "min" else 0
    self._values = dict( (v, i + d) for (i, v) in enumerate(values) )
    
  def size( self ):
    return len(self._values) + (1 if self._default is not None else 0)
    
  def __call__( self, s ):
    i = self._values.get( self._feature(s), None )
    if self._default is not None:
      if self._default == "min":
        return 0 if i is None else i
      else:
        return self.size() - 1 if i is None else i
    elif i is None:
      raise ValueError( "No mapping for {} and use_default = False".format( s ) )
    else:
      return i
      
  def inverse( self, f ):
    for (k, v) in self._values.items():
      if v == f:
        return k
    return None
    # raise ValueError( "No (unique) inverse for {}".format( f ) )
    
class Binned(Feature):
  def __init__( self, feature, ranges, name=None ):
    self._name = "Binned({})".format(feature.name()) if not name else name
    self._feature = feature
    self._ranges = ranges

  def size( self ):
    return len(self._ranges)
    
  def __call__( self, s ):
    f = self._feature( s )
    for (i, (lower, upper)) in enumerate(self._ranges):
      if lower <= f and f < upper:
        return i
    raise ValueError( "No bin for {}".format( s ) )

def binned_contiguous( feature, breaks, lower=-math.inf, upper=math.inf, name=None ):
  breaks = [lower] + breaks + [upper]
  ranges = []
  for i in range(1, len(breaks)):
    ranges.append( (breaks[i-1], breaks[i]) )
  return Binned(feature=feature, ranges=ranges, name=name)
  
# ----------------------------------------------------------------------------

class FiniteFeatureSpace:
  def __init__( self, features ):
    self._features = features
    
  def feature( self, i ):
    return self._features[i]
  
  def encode( self, s ):
    return tuple( f(s) for f in self._features )
  
  def states( self ):
    idx = [0] * len(self._features)
    while True:
      # FIXME: This only supports packed features in {0, 1, ..., n}
      yield tuple( idx[:] )
      for i in range(len(idx)):
        # Increment LSB
        idx[i] += 1
        if idx[i] < self._features[i].size():
          break
        else:
          # Ripple-carry
          idx[i] = 0
      # Terminate if we wrapped back to [0, 0, ..., 0]
      if all( i == 0 for i in idx ):
        break
