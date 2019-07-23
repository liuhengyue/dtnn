import statistics

class MeanAccumulator:
  def __init__( self ):
    self._mean = 0
    self._n = 0
  
  def mean( self ):
    return self._mean
    
  def n( self ):
    return self._n
  
  def __call__( self, x ):
    self._n    += 1
    self._mean += (x - self._mean) / self._n

class MovingMeanAccumulator:
  def __init__( self, window ):
    self._samples = [0] * window
    self._i = 0
    self._mean = 0
    self._n = 0
    
  def mean( self ):
    return self._mean
    
  def n( self ):
    return self._n
    
  def __call__( self, x ):
    window = len(self._samples)
    self._n += 1
    if self._n <= window:
      # Don't include initial "dummy" values in the mean
      self._mean += (x - self._mean) / self._n
    else:
      # Replace old value
      old = self._samples[self._i]
      self._mean += (x - old) / window
    self._samples[self._i] = x
    self._i = (self._i + 1) % window
