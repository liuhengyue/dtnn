""" Classes for package-agnostic neural network architecture specifications.

There are two kinds of classes pertaining to architecture specification: 
`Spec`s and `SpecSet`s. The avaliable `Spec`s are enumerated in `LayerSpec`.
Each one holds the parameters of a neural net layer. Each `Spec` type has a
corresponding `SpecSet` type that generates `Spec` instances from the cross
product of the provided parameter lists. A `SpecSet` is initialized with the
same named parameters as the corresponding `Spec`, but each parameter is a
`list` instead of a singleton.

A `SpecSet` represents the types of layers that can be added by the search
algorithm. A complete search space also requires a `SearchStrategy` and a
state evaluation `Heuristic`. The strategy tells the search where it can insert
new layers, and the heuristic estimates the value of unexplored network
architectures.
"""

from collections import namedtuple
import enum
import functools
import itertools
import re

# ----------------------------------------------------------------------------

ArchLayerString = namedtuple( "ArchLayerString", ["inputs", "body", "alias"] )

def join_arch_strings( strings ):
  return ":".join( strings )
  
def split_arch_string( s ):
  return s.split( ":" )
  
def decompose_arch_layer_string( s ):
  # FIXME: Doesn't support [-] inputs (only [+] inputs)
  #                        inputs     body   alias
  pattern = re.compile( r"(\[\+([^\]]+)\])?([^\(]+)(\(([^\)]+)\))?" )
  m = re.match( pattern, s )
  input_group = m.group( 2 )
  inputs = input_group.split( "," ) if input_group is not None else []
  body = m.group( 3 )
  alias_group = m.group( 5 )
  alias = alias_group.split( "-" )[1] if alias_group is not None else None
  return ArchLayerString( inputs, body, alias )
  
def join_layer_params( name, *params ):
  def impl( *tokens ):
    return "-".join( map( str, tokens ) )
  return impl( name, *params )
  
def split_layer_params( s ):
  return s.split( "-" )
    
# ----------------------------------------------------------------------------

class LayerSpecBase:
  def __init__( self, alias=None, inputs=[] ):
    self._alias = alias
    self._inputs = inputs
  
  def alias( self ):
    return self._alias
  
  def inputs( self ):
    return self._inputs[:]

  def __eq__( self, other ):
    return self.arch_string() == other.arch_string()
    
  def __hash__( self ):
    return hash(self.arch_string())
    
  def arch_string( self ):
    if len(self._inputs) > 0:
      input_aliases = [i.alias() for i in self._inputs]
      assert( all( a is not None for a in input_aliases ) )
      prefix = "[+{}]".format( ",".join( input_aliases ) )
    else:
      prefix = ""
    if self._alias is not None:
      suffix = "(alias-{})".format( self._alias )
    else:
      suffix = ""
    return self._arch_string_impl().join( [prefix, suffix] )

class InputLayerSpec(LayerSpecBase):
  @staticmethod
  def arch_prefix():
    return "i"
  
  def __init__( self, c, h, w, alias=None, inputs=[] ):
    """
    Parameters:
      c : Number of channels
      h : Height
      w : Width
    """
    if len(inputs) > 0:
      raise ValueError( "extra inputs not allowed" )
    super().__init__( alias=alias )
    self._input_dim = (c, h, w)
    
  def channels( self ):
    return self._input_dim[0]
  
  def height( self ):
    return self._input_dim[1]
  
  def width( self ):
    return self._input_dim[2]
    
  def _arch_string_impl( self ):
    return join_layer_params( self.__class__.arch_prefix(), *self._input_dim )

class OutputLayerSpec(LayerSpecBase):
  @staticmethod
  def arch_prefix():
    return "l"
    
  @staticmethod
  def SetType():
    return OutputLayerSet
  
  def __init__( self, nclasses, alias=None, inputs=[] ):
    """
    Parameters:
      nclasses : Number of possible class labels
    """
    super().__init__( alias=alias, inputs=inputs )
    self._nclasses = nclasses
    
  def _arch_string_impl( self ):
    return join_layer_params( self.__class__.arch_prefix(), self._nclasses )

class OutputLayerSet:
  def __init__( self, nclasses ):
    assert( len(nclasses) == 1 )
    self._nclasses = nclasses
  
  def __iter__( self ):
    yield OutputLayerSpec( self._nclasses[0] )

# ----------------------------------------------------------------------------

class ConvolutionLayerSpec(LayerSpecBase):
  @staticmethod
  def arch_prefix():
    return "conv"
    
  @staticmethod
  def SetType():
    return ConvolutionLayerSet
  
  def __init__( self, filters, filter_size, xstride=1, ystride=1, alias=None, inputs=[] ):
    """
    Parameters:
      filters : Number of filters
      filter_size : Width/height of filters (always square)
      xstride : Stride in x-direction
      ystride : Stride in y-direction
    """
    super().__init__( alias=alias, inputs=inputs )
    self._filters = filters
    self._filter_size = filter_size
    self._xstride = xstride
    self._ystride = ystride
  
  def filters( self ):
    return self._filters
  
  def filter_size( self ):
    return self._filter_size
    
  def _arch_string_impl( self ):
    return join_layer_params(
      self.__class__.arch_prefix(), self._filters, self._filter_size, self._xstride, self._ystride )
  
class ConvolutionLayerSet:
  """ Generator yielding combinations of convolution layer parameters.
  """
  def __init__( self, *, filters, filter_size, xstride, ystride ):
    """
    Parameters:
      filters : List of numbers of filters
      filter_size : List of filter sizes
      xstride : List of x-direction strides
      ystride : List of y-direction strides
    """
    self._filters = filters
    self._filter_size = filter_size
    self._xstride = xstride
    self._ystride = ystride
    
  def __iter__( self ):
    for r in self._filters:
      for s in self._filter_size:
        for t in self._xstride:
          for u in self._ystride:
            yield ConvolutionLayerSpec( r, s, t, u )

# ----------------------------------------------------------------------------

@enum.unique
class ActivationFunction(enum.Enum):
  """ String identifiers of implemented activation functions.
  """
  tanh = "tanh"
  relu = "relu"
  lrelu = "lrelu"

class ActivationLayerSpec(LayerSpecBase):
  @staticmethod
  def arch_prefix():
    return "act"
    
  @staticmethod
  def SetType():
    return ActivationLayerSet
  
  def __init__( self, fn, p=None, alias=None, inputs=[] ):
    """
    Parameters:
      fn : ActivationFunction or string
      p : If fn is lrelu, "leak" parameter in [0,1]. Otherwise should be `None`
    """
    super().__init__( alias=alias, inputs=inputs )
    self._fn = ActivationFunction(fn)
    self._p = p
    assert( not self._p or self._fn is ActivationFunction.lrelu )
  
  def _arch_string_impl( self ):
    s = [self.__class__.arch_prefix(), self._fn.value]
    if self._p: s.append( self._p )
    return join_layer_params( *s )

class ActivationLayerSet:
  def __init__( self, *, fn, lrelu=[] ):
    """
    Parameters:
      tanh : bool. If True, include `tanh` activation.
      relu : bool. If True, include `relu` activation.
      lrelu : [float]. If non-empty, include `lrelu` activation for each
        provided value of "leak" parameter.
    """
    self._specs = []
    fn = [ActivationFunction(f) for f in fn]
    if ActivationFunction.tanh in fn: self._specs.append( ActivationLayerSpec( ActivationFunction.tanh ) )
    if ActivationFunction.relu in fn: self._specs.append( ActivationLayerSpec( ActivationFunction.relu ) )
    for p in lrelu:
      assert( ActivationFunction.lrelu in fn )
      self._specs.append( ActivationLayerSpec( ActivationFunction.lrelu, p ) )
      
  def __iter__( self ):
    return iter(self._specs)

# ----------------------------------------------------------------------------

class PoolLayerSpec(LayerSpecBase):
  @staticmethod
  def arch_prefix():
    return "pool"
    
  @staticmethod
  def SetType():
    return PoolLayerSet
  
  def __init__( self, fn, xpool, ypool, xstride, ystride, alias=None, inputs=[] ):
    """
    Parameters:
      fn : Pool function
      xpool : Pool width in x-direction
      ypool : Pool width in y-direction
      xstride : Pool stride in x-direction (usually equal to width)
      ystride : Pool stride in y-direction (usually equal to width)
    """
    super().__init__( alias=alias, inputs=inputs )
    self._fn = fn
    self._xpool = xpool
    self._ypool = ypool
    self._xstride = xstride
    self._ystride = ystride
    
  def xpool( self ):
    return self._xpool
    
  def ypool( self ):
    return self._ypool
    
  def _arch_string_impl( self ):
    return join_layer_params( self.__class__.arch_prefix(), self._fn,
      self._xpool, self._ypool, self._xstride, self._ystride )

class PoolLayerSet:
  def __init__( self, *, fn, xpool, ypool, xstride, ystride ):
    self._fn = fn
    self._xpool = xpool
    self._ypool = ypool
    self._xstride = xstride
    self._ystride = ystride
  
  def __iter__( self ):
    for f in self._fn:
      for r in self._xpool:
        for s in self._ypool:
          for t in self._xstride:
            for u in self._ystride:
              yield PoolLayerSpec( f, r, s, t, u )
              
# ----------------------------------------------------------------------------

# Inheritance here is not good OOP, but needed to allow
# isinstance(PoolLayerSpec) to detect all pool layers
class SquarePoolLayerSpec(PoolLayerSpec):
  @staticmethod
  def arch_prefix():
    return "pool"
    
  @staticmethod
  def SetType():
    return SquarePoolLayerSet
  
  def __init__( self, fn, pool, stride, alias=None, inputs=[] ):
    """
    Parameters:
      fn : Pool function
      pool : Pool width
      stride : Pool stride
    """
    super().__init__( fn, pool, pool, stride, stride, alias=alias, inputs=inputs )

class SquarePoolLayerSet:
  def __init__( self, *, fn, pool, stride ):
    self._fn = fn
    self._pool = pool
    self._stride = stride
  
  def __iter__( self ):
    for f in self._fn:
      for p in self._pool:
        for s in self._stride:
          yield PoolLayerSpec( f, p, p, s, s )

# ----------------------------------------------------------------------------

class FullyConnectedLayerSpec(LayerSpecBase):
  @staticmethod
  def arch_prefix():
    return "fc"
    
  @staticmethod
  def SetType():
    return FullyConnectedLayerSet
  
  def __init__( self, n, alias=None, inputs=[] ):
    super().__init__( alias=alias, inputs=inputs )
    self._n = n
    
  def n( self ):
    return self._n
    
  def _arch_string_impl( self ):
    return join_layer_params( self.__class__.arch_prefix(), self._n )

class FullyConnectedLayerSet:
  def __init__( self, n ):
    self._n = n
    
  def __iter__( self ):
    for n in self._n:
      yield FullyConnectedLayerSpec( n )

# ----------------------------------------------------------------------------

class DropoutLayerSpec(LayerSpecBase):
  @staticmethod
  def arch_prefix():
    return "drop"
    
  @staticmethod
  def SetType():
    return DropoutLayerSet

  def __init__( self, p, alias=None, inputs=[] ):
    super().__init__( alias=alias, inputs=inputs )
    self._p = p
    
  def _arch_string_impl( self ):
    return join_layer_params( self.__class__.arch_prefix(), self._p )

class DropoutLayerSet:
  def __init__( self, p ):
    self._p = p
    
  def __iter__( self ):
    for p in self._p:
      yield DropoutLayerSpec( p )

# ----------------------------------------------------------------------------

class BatchNormLayerSpec(LayerSpecBase):
  @staticmethod
  def arch_prefix():
    return "bnorm"
  
  @staticmethod
  def SetType():
    return BatchNormLayerSet
    
  def __init__( self, alias=None, inputs=[] ):
    super().__init__()
  
  def _arch_string_impl( self ):
    return self.__class__.arch_prefix()

class BatchNormLayerSet:
  def __iter__( self ):
    yield BatchNormLayerSpec()

# ----------------------------------------------------------------------------

class SequenceSpec(LayerSpecBase):
  """ Specification of a sequence of layers.
  """
  @staticmethod
  def SetType():
    return SequenceSet
    
  def __init__( self, *specs, alias=None, inputs=[] ):
    super().__init__( alias=alias, inputs=inputs )
    self._specs = specs
    
  def flat( self ):
    f = []
    for spec in self:
      if isinstance(spec, SequenceSpec):
        f.extend( spec.flat() )
      else:
        f.append( spec )
    return f
    
  def learnable( self ):
    return [s for s in self.flat() if isinstance(s, ConvolutionLayerSpec) 
            or isinstance(s, FullyConnectedLayerSpec)]
            
  def has_fc_layer( self ):
    return any( isinstance(layer, FullyConnectedLayerSpec) for layer in self.flat() )
    
  def get_fc_layers( self ):
    fc_layers = []
    for layer in self.flat():
      if isinstance(layer, FullyConnectedLayerSpec):
        fc_layers.append( layer )
    return fc_layers
    
  def has_conv_layer( self ):
    return any( isinstance(layer, ConvolutionLayerSpec) for layer in self.flat() )
    
  def get_conv_layers( self ):
    return [layer for layer in self.flat() if isinstance(layer, ConvolutionLayerSpec)]
    
  def has_pool_layer( self ):
    return any( isinstance(layer, PoolLayerSpec) for layer in self.flat() )
    
  def get_pool_layers( self ):
    return [layer for layer in self.flat() if isinstance(layer, PoolLayerSpec)]
  
  def has_input_layer( self ):
    return isinstance(self.flat()[0], InputLayerSpec)
    
  def get_input_layer( self ):
    return [layer for layer in self.flat() if isinstance(layer, InputLayerSpec)]
  
  def has_output_layer( self ):
    return any( isinstance(layer, OutputLayerSpec) for layer in self.flat() )
  
  def features( self ):
    fc = self.get_fc_layers()
    if fc:
      return fc[-1].n()
    conv = self.get_conv_layers()
    if conv:
      return conv[-1].filters()
    input = self.get_input_layer()
    if input:
      return input[-1].channels()
    return 0
    
  def receptive_field( self ):
    """ Approximately the square root of the number of input units to each
    output unit (e.g. side length of a convolutional filter or pool operation).
    
    A value of `0` represents the entire input (i.e. fully connected layers).
    """
    conv = self.get_conv_layers()
    if conv:
      return conv[-1].filter_size()
    pool = self.get_pool_layers()
    if pool:
      return max( pool[-1].xpool(), pool[-1].ypool() )
    return 0 # Represents "whole input"

  def __len__( self ):
    return len(self._specs)
    
  def __getitem__( self, i ):
    return self._specs[i]
    
  def __iter__( self ):
    return iter(self._specs)
    
  def _arch_string_impl( self ):
    return join_arch_strings( list(spec.arch_string() for spec in self._specs) )
    
  def __str__( self ):
    # return "Sequence({})".format( [str(s) for s in self] )
    return self.arch_string()
    
  # def __hash__( self ):
    # return hash( self.arch_string() )
    
  # def __eq__( self, other ):
    # return self.arch_string() == other.arch_string()

class SequenceSet:
  def __init__( self, *layer_sets ):
    self._layer_sets = layer_sets
    
  def __iter__( self ):
    for layer_specs in itertools.product( *self._layer_sets ):
      yield SequenceSpec( *layer_specs )

# ----------------------------------------------------------------------------

@enum.unique
class LayerSpec(enum.Enum):
  """ Available layer types.
  """
  Input           = InputLayerSpec
  Output          = OutputLayerSpec
  Convolution     = ConvolutionLayerSpec
  Activation      = ActivationLayerSpec
  Pool            = PoolLayerSpec
  SquarePool      = SquarePoolLayerSpec
  FullyConnected  = FullyConnectedLayerSpec
  Dropout         = DropoutLayerSpec
  BatchNorm       = BatchNormLayerSpec
  Sequence        = SequenceSpec
  
  # def __call__( self, *args, **kwargs ):
    # return self.value( *args, **kwargs )
  
  @staticmethod
  def from_arch_string( s, aliases=dict() ):
    """ Creates a LayerSpec from a string in the format consumed by Architect.
    """
    layer_strings = split_arch_string( s )
    if len(layer_strings) > 1:
      layers = [LayerSpec.from_arch_string( ls, aliases ) for ls in layer_strings]
      return SequenceSpec( *layers )
    else:
      archstring = decompose_arch_layer_string( s )
      print( str(archstring) )
      inputs = [aliases[k] for k in archstring.inputs] if archstring.inputs is not None else []
      tokens = split_layer_params( archstring.body )
      print( str(tokens) )
      arch_prefix = tokens[0]
      tokens = tokens[1:]
      for layer_type in LayerSpec:
        if arch_prefix == layer_type.value.arch_prefix():
          layer = layer_type.value( *tokens, alias=archstring.alias, inputs=inputs )
          if archstring.alias is not None:
            aliases[archstring.alias] = layer
          return layer
      raise RuntimeError( "Invalid layer type in '" + s + "'" )
      
# ----------------------------------------------------------------------------

def _parse_json_layer( entry, set_context=False ):
  """ Parse a JSON dict specifying a single network layer.
  
  Parameters:
    set_context : bool. If `True`, interpret `entry` as a `LayerSet` rather
      than a `LayerSpec`
  """
  spec_type = LayerSpec[entry["layer"]]
  parameters = entry["parameters"]
  if type(parameters) is list:
    assert( spec_type is LayerSpec.Sequence )
    subspecs = [_parse_json_layer( sub, set_context ) for sub in entry["parameters"]]
    if set_context:
      spec = spec_type.value.SetType()( *subspecs )
    else:
      spec = spec_type.value( *subspecs )
  else:
    if set_context:
      p = entry["parameters"]
      d = {k : [p[k]] if type(p[k]) is not list else p[k] for k in p}
      spec = spec_type.value.SetType()( **d )
    else:
      spec = spec_type.value( **entry["parameters"] )
  return spec

def parse_json_architecture( json_list ):
  """ Parse a JSON list of `LayerSpec`s that should be treated as a full
  network specification.
  """
  return SequenceSpec( *[_parse_json_layer( layer ) for layer in json_list] )
  
def parse_json_layer_set( json_list ):
  """ Parse a JSON list of `LayerSpec`s that should be treated as a single
  module.
  """
  return [layer for lset in json_list for layer in _parse_json_layer( lset, set_context=True )]

def parse_json_search_space( json_dict ):
  """ Parse a JSON dict containing a single search space specification.
  """
  d = dict()
  
  d["input"] = _parse_json_layer( json_dict["input"] )
  # d["output"] = _parse_json_layer( json_dict["output"] )
  d["output"] = parse_json_layer_set( json_dict["output"] )
  d["layers"] = parse_json_layer_set( json_dict["layers"] )
  # d["actions"] = [Actions.from_json( json_a, layer )
                  # for json_a in json_dict["actions"] 
                  # for layer_set in d["layers"]
                  # for layer in layer_set]
  # d["actions"].insert( 0, TrainAndEvaluateAction() )
  # )SearchStrategy[json_dict["strategy"]].value
  # d["heuristic"] = Heuristic.from_json( json_dict["heuristic"] )
  # Patch heuristic function into 'instance' as a method
  # instance.heuristic = types.MethodType( lambda _self, s: heuristic(s), instance )
  # return instance
  return d
  
def parse_json_search_space_list( json_list ):
  """ Parse a JSON list of search space specifications.
  """
  search_spaces = []
  for search_space in json_list:
    search_spaces.append( parse_json_search_space( search_space ) )
  return search_spaces
