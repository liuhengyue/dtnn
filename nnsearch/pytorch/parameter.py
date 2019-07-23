import abc
import argparse
import math

class Hyperparameter(metaclass=abc.ABCMeta):
  def __init__( self, name ):
    self._name = name
  
  @abc.abstractmethod
  def __call__( self ):
    return NotImplemented
  
  @property
  def name( self ):
    return self._name

  def set_batch( self, batch_idx ):
    pass
    
  def set_epoch( self, epoch_idx, nbatches ):
    pass
    
  def __str__( self ):
    return "{}: {} [{}]".format(self.name, self(), self._str())
    
  def _str( self ):
    return ""
    
class Transform(Hyperparameter):
  def __init__( self, base, transform, name=None ):
    self._base = base
    self._transform = transform
    self._name = name
    
  def __call__( self ):
    return self._transform( self._base() )
  
  @property
  def name( self ):
    if self._name is not None:
      return self._name
    else:
      return self._base.name
  
  def set_batch( self, batch_idx ):
    self._base.set_batch( batch_idx )
    
  def set_epoch( self, epoch_idx, nbatches ):
    self._base.set_epoch( epoch_idx, nbatches )
    
  def _str( self ):
    return self._base._str()

class Constant(Hyperparameter):
  def __init__( self, name, value ):
    super().__init__( name )
    self._value = value
    
  def __call__( self ):
    return self._value
    
  def _str( self ):
    return "Constant({})".format(self._value)
    
class LinearSchedule(Hyperparameter):
  def __init__( self, name, start, increment, stop, delta_t=1 ):
    super().__init__( name )
    assert( math.copysign(start, increment) <= stop )
    self._start = start
    self._increment = increment
    self._stop = stop
    self._delta_t = delta_t
    self._epoch = None
    
  def __call__( self ):
    step = self._epoch // self._delta_t
    d = self._increment * step
    if abs(d) > abs(self._start - self._stop):
      return self._stop
    else:
      return self._start + d
    
  def set_epoch( self, epoch_idx, nbatches ):
    self._epoch = epoch_idx
    
  def _str( self ):
    return "Linear({}, {}, {}, {})".format(self._start, self._increment,
      self._stop, self._delta_t)
    
class GeometricSchedule(Hyperparameter):
  def __init__( self, name, start, factor ):
    super().__init__( name )
    self._start = start
    self._factor = factor
    self._epoch = None
    
  def __call__( self ):
    return self._start * (self._factor ** self._epoch)
    
  def set_epoch( self, epoch_idx, nbatches ):
    self._epoch = epoch_idx
    
  def _str( self ):
    return "Geometric({}, {})".format(self._start, self._factor)
    
class HarmonicSchedule(Hyperparameter):
  def __init__( self, name, base, exponent, delta_t=1 ):
    super().__init__( name )
    self._base = base
    self._exponent = exponent
    self._delta_t = delta_t
    self._epoch = None
    
  def __call__( self ):
    step = self._epoch // self._delta_t
    t = step + 1
    if self._exponent == 1:
      return self._base / t
    else:
      return self._base / math.pow( t, self._exponent )
      
  def set_epoch( self, epoch_idx, nbatches ):
    self._epoch = epoch_idx
    
  def _str( self ):
    return "Harmonic({}, {}, {})".format(
      self._base, self._exponent, self._delta_t)
    
class StepSchedule(Hyperparameter):
  def __init__( self, name, values, times ):
    assert( len(values) == len(times) + 1 )
    super().__init__( name )
    self._values = values
    self._times = times
    self._epoch = None
    
  def __call__( self ):
    for (i, t) in enumerate(self._times):
      if t > self._epoch:
        return self._values[i]
    return self._values[-1]
  
  def set_epoch( self, epoch_idx, nbatches ):
    self._epoch = epoch_idx
    
  def _str( self ):
    return "Step({}, {})".format(self._values, self._times)
    
class CosineSchedule(Hyperparameter):
  def __init__( self, name, maxval, minval, Ti, Tmult ):
    super().__init__( name )
    assert( maxval >= minval )
    assert( Ti > 0 )
    assert( Tmult > 0 )
    self._maxval = maxval
    self._minval = minval
    self._Ti_init = Ti
    self._Tmult = Tmult
    self._epoch = None
    self._nbatches = None
    self._e0 = None
    self._Ti = None
    self._progress = 0
    
  def __call__( self ):
    m = 0.5 * (self._maxval - self._minval)
    return self._minval + m * (1 + math.cos(math.pi * self._progress))
  
  def _stage_info( self ):
    """ We have to jump through some hoops to allow the implementation to be
    reset to arbitrary epochs. Instead of accumulating the values of internal
    parameters over time, this function computes them for the current epoch.
    
    Returns:
      `(e0, Ti)` : where `e0` is the epoch when the current "stage" began, and
        `Ti` is the value of `Ti` during the current stage.
    """
    def restarts():
      Ti = self._Ti_init
      e0 = 0
      while True:
        yield (e0, Ti)
        e0 += Ti
        Ti *= self._Tmult
    last = None
    for (e0, Ti) in restarts():
      if e0 <= self._epoch:
        last = (e0, Ti)
      else:
        break
    return last
  
  def set_epoch( self, epoch_idx, nbatches ):
    self._epoch = epoch_idx
    self._nbatches = nbatches
    self._e0, self._Ti = self._stage_info()
    self.set_batch( 0 )
    
  def set_batch( self, batch_idx ):
    Tcur = (self._epoch - self._e0) + (batch_idx / self._nbatches)
    assert( 0 <= Tcur < self._Ti )
    self._progress = Tcur / self._Ti
    
  def _str( self ):
    return "Cosine({}, {}, {}, {})".format(
      self._maxval, self._minval, self._Ti_init, self._Tmult )

# ----------------------------------------------------------------------------
# Cmd parsing

def _schedule_from_spec( name, spec ):
  tokens = spec.split( "," )
  if tokens[0] == "constant":
    return Constant( name, float(tokens[1]) )
  elif tokens[0] == "linear":
    return LinearSchedule( name, *map(float, tokens[1:]) )
  elif tokens[0] == "geometric":
    return GeometricSchedule( name, *map(float, tokens[1:]) )
  elif tokens[0] == "harmonic":
    return HarmonicSchedule( name, *map(float, tokens[1:]) )
  elif tokens[0] == "step":
    xs = [float(r) for r in tokens[1::2]]
    times = [int(t) for t in tokens[2::2]]
    return StepSchedule( name, xs, times )
  elif tokens[0] == "cosine":
    return CosineSchedule( name,
      float(tokens[1]), float(tokens[2]), int(tokens[3]), int(tokens[4]) )
  else:
    raise ValueError( "invalid parameter schedule spec: '{}'".format(spec) )
    
def schedule_spec( name ):
  """ Returns `schedule_spec` with the first parameter bound to `name`. The
  result is suitable for use as a `type` function in `argparse`.
  
  It is easier to use `parse_schedule()` as an `action`, but that approach
  cannot be used with default values, since they are never passed to the
  action handler.
  """
  def f( spec ):
    return _schedule_from_spec( name, spec )
  return f
        
def parse_schedule():
  """ Returns an `argparse.Action` that handles hyperparameter schedule
  specifications.
  
  This is the easiest way to parse schedules with `argparse`, but it cannot be
  used with default values, since they are never passed to the action handler.
  Use `schedule_arg(name)` as the `type` parameter if you need defaults.
  """
  class ScheduleParser(argparse.Action):
    def __init__( self, *args, **kwargs ):
      super().__init__( *args, **kwargs )

    def __call__( self, parser, namespace, values, option_string=None ):
      if values is None:
        setattr(namespace, self.dest, None)
        return
      try:
        p = _schedule_from_spec( option_string, values )
      except ValueError:
        parser.error( option_string )
      setattr(namespace, self.dest, p)
      
  return ScheduleParser
      
# ----------------------------------------------------------------------------

if __name__ == "__main__":
  cos = CosineSchedule( "learning_rate", 0.05, 0, 2, 2 )
  nbatches = 5
  nepochs = 7
  for e in range(nepochs):
    print( "epoch: {}".format(e) )
    cos.set_epoch( e, nbatches )
    for b in range(nbatches):
      cos.set_batch( b )
      print( "e0: {}; Ti: {}; p: {}".format(cos._e0, cos._Ti, cos()) )
