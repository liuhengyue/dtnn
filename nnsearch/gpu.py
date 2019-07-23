import logging
import os.path
import re
import shlex
import subprocess

import nnsearch.objective as objective

class GpuLogger:
  """ Uses the `nvidia-smi` command line tool to collect power usage data from
  the GPU.
  
  *Should* work on all platforms, but I can only test on Linux.
  """
  query_string = "index,timestamp,power.draw,temperature.gpu,utilization.gpu,utilization.memory"

  def __init__( self, log_dir ):
    """
    Parameters:
      `log_dir` : Directory to store log files
      `gpu_name` : Unique identifier (not 0-based index!) of the GPU to log
    """
    self._log_dir = log_dir
    self._proc = dict()
    self._idle_power = 18.0 #W
    self._sample_interval_ms = 250
    assert( 1000 % self._sample_interval_ms == 0 )
    self._sample_Hz = 1000 // self._sample_interval_ms
    
  def _filename( self, gpu_id ):
    return os.path.join( self._log_dir, "nvidia-smi_{}.log".format( gpu_id ) )
    
  def start( self, gpu_id ):
    """ Start logging. Launches a subprocess that logs data to a file.
    
    Parameters:
      `gpu_id` : Identifier (UID or index) of the GPU to log
    """
    if gpu_id in self._proc:
      logging.getLogger( __name__ ).warn( "already started" )
    else:
      cmd = "nvidia-smi --id={} --query-gpu={} --format=csv --loop-ms={}".format(
        gpu_id, GpuLogger.query_string, self._sample_interval_ms )
      logging.getLogger( __name__ ).info( "Starting GpuLogger: {}".format( cmd ) )
      args = shlex.split( cmd )
      with open( self._filename( gpu_id ), "w" ) as fout:
        proc = subprocess.Popen( args, stdout=fout, stderr=subprocess.DEVNULL )
      self._proc[gpu_id] = proc
  
  def stop( self, gpu_id ):
    """ Stop logging. Kills the subprocess then processes the data file.
    
    Parameters:
      `gpu_id` : Identifier (UID or index) of the GPU to log
    
    Return
      `(total_energy, active_energy)` : `total_energy` is energy consumption
        excluding idle periods at the beginning and end. `active_energy`
        excludes *all* idle periods (`active_energy <= total_energy`).
    """
    proc = self._proc[gpu_id]    
    proc.kill()
    proc.wait()
    del self._proc[gpu_id]
    powers = self._get_power_measurements( gpu_id )
    powers = self._trim_idle( powers )
    total_energy  = sum( powers ) / self._sample_Hz
    active_energy = sum( p for p in powers if p > self._idle_power ) / self._sample_Hz
    return objective.Energy(total=total_energy, active=active_energy)
    
  def shutdown( self ):
    for proc in self._proc.values():
      try:
        proc.kill()
      except:
        pass
    self._proc = dict()
  
  def _get_power_measurements( self, gpu_id ):
    powers = []
    with open( self._filename( gpu_id ), "r" ) as fin:
      number = re.compile( "([+-]?\d*\.\d+|\d+|inf|nan)" )
      for (i, line) in enumerate(fin):
        if i == 0: # Skip header
          continue
        power_str = line.split( "," )[2]
        power = float( re.search( number, power_str ).group(0) )
        powers.append( power )
    return powers
    
  def _trim_idle( self, powers ):
    first = None
    last = len(powers) - 1
    for (i, p) in enumerate(powers):
      if p > self._idle_power:
        if first is None:
          first = i
        last = i
    if first is None:
      return []
    return powers[first:(last+1)]

if __name__ == "__main__":
  import sys
  import time
  gpu_id = sys.argv[1]
  directory = sys.argv[2]
  gpu_logger = GpuLogger( directory )
  gpu_logger.start( gpu_id )
  time.sleep( 10 )
  energy = gpu_logger.stop( gpu_id )
  print( energy )
  
