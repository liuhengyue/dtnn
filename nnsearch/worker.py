import concurrent.futures as futures
import logging
import os
import os.path
import time

import nnsearch.backend.training
import nnsearch.objective as objective

# ----------------------------------------------------------------------------

class Train(nnsearch.backend.training.ControllerBase):
  def __init__( self, nepoch = 1 ):
    self._nepoch = nepoch
    self._epoch = 0
    
  def begin_epoch( self, model ):
    return self._epoch < self._nepoch
  
  def finish_epoch( self, model ):
    self._epoch += 1

class Pool:
  """ Performs neural network training and evaluation in a process pool.
  """
  
  log_format = "[%(process)d]:%(levelname)s:%(name)s:%(message)s"
  
  @staticmethod
  def _get_logger():
    pid = os.getpid()
    return logging.getLogger( str(pid) + ":" + __name__ )

  @staticmethod
  def _train_and_evaluate( step, storage, backend, parameters, s, nepoch, gpu_ids,
                           validate=True, test=False, gpu_logger=None ):
    gpu_id = gpu_ids[os.getpid()]
    log = logging.getLogger( __name__ )
    log.info( "STEP %s (%s)", step, gpu_id if gpu_id is not None else "cpu" )
    if gpu_logger is not None:
      log.info( "Using GpuLogger" )
    
    # FIXME: Make string encoding backend-agnostic
    nn_string = s.arch_string()
    log.info( "Model %s", nn_string )
    filename = storage.get_filename( nn_string )
    if storage.contains( nn_string ):
      log.info( "Loading from %s", filename )
      nn = backend.load_model( filename, parameters )
    else:
      nn = backend.create_model( parameters, s )
    
    controller = Train( nepoch=nepoch )
    log.info( "Training" )
    nn.train( controller )
    
    # Default return values
    (validation_acc, test_acc, energy) = (None, None, None)
    if gpu_logger is not None and gpu_id is not None:
      gpu_logger.start( gpu_id )
    if validate:
      log.info( "Validating" )
      validation_acc = 1.0 - nn.validate()
      log.info( "validation_accuracy=%s", validation_acc )
    if test:
      log.info( "Testing" )
      test_acc = 1.0 - nn.test()
      log.info( "test_accuracy=%s", test_acc )
    if gpu_logger is not None:
      energy = gpu_logger.stop( gpu_id )
      log.info( "energy=%s", energy )
    
    log.info( "Saving to %s", filename )
    nn.save_model( filename )
    return objective.Performance( objective.Accuracy(validation_acc, test_acc), energy )
    
  @staticmethod
  def _init_worker( backend, logdir, log_level, gpu_id=None, gpu_name=None ):
    logfile = os.path.join( logdir, "output_pid{}.log".format( os.getpid() ) )
    
    # log = Pool._get_logger()
    # handler = logging.FileHandler( logfile, "w" )
    # handler.setLevel( log_level )
    # formatter = logging.Formatter( Pool.log_format )
    # handler.setFormatter( formatter )
    # log.addHandler( handler )
    
    logging.basicConfig( filename=logfile, filemode="w",
                         format=Pool.log_format, level=log_level )
    log = logging.getLogger( __name__ )
                         
    log.info( "Worker is alive (pid: %s)", os.getpid() )
    log.info( "gpu_id=%s,gpu_name=%s", gpu_id, gpu_name )
    
    log.info( "[%s] Environment:", os.getpid() )
    for k in sorted( os.environ ):
      log.info( "%s: %s", k, os.environ[k] )

    if gpu_id is not None:
      backend.gpu_setup( gpu_id, gpu_name )
    # Pause for a while to make sure that subsequent setup calls are dispatched
    # to different processes
    time.sleep( 5 )
    return (os.getpid(), gpu_id)
  
  def __init__( self, storage, backend, parameters, nworkers, gpu_workers=[], gpu_names=None ):
    """
    Parameters:
      `base` : A search space over `LayerSpec`s
    """
    self._storage = storage
    self._backend = backend
    self._parameters = parameters
    self._pool = futures.ProcessPoolExecutor( max_workers = nworkers )
    
    log = logging.getLogger( __name__ )
    
    # Pool setup
    fs = []
    for i in range(nworkers):
      gpu_id = gpu_workers[i] if i < len(gpu_workers) else None
      if gpu_names is not None and gpu_id is not None:
        gpu_name = gpu_names[i]
      else:
        gpu_name = None
      # FIXME: Relying on fields of 'parameters'
      f = self._pool.submit( Pool._init_worker,
        self._backend, self._parameters.output, self._parameters.level, gpu_id, gpu_name )
      fs.append( f )
    futures.wait( fs )
    # Build mapping from PID to GPU identifier
    if gpu_names is not None:
      assign = [f.result() for f in fs]
      self._gpu_assignments = dict()
      for (i, (pid, id)) in enumerate(assign):
        if id is not None:
          self._gpu_assignments[pid] = gpu_names[i]
        else:
          self._gpu_assignments[pid] = None
    else:
      log.warn( "No 'gpu_names' provided; using numeric IDs. IDs may not refer to the same GPU across runs." )
      self._gpu_assignments = dict( [f.result() for f in fs] )
    if len(self._gpu_assignments) != nworkers:
      log.error(
        "Some workers were not initialized: nworkers = %s, gpu_workers = %s, gpu_assignments = %s",
        nworkers, gpu_workers, self._gpu_assignments )
      raise RuntimeError( "Some workers were not initialized" )
    log.info( "GPU assignments: %s", str(self._gpu_assignments) )
    
    self._step = 0
    
  def shutdown( self ):
    self._pool.shutdown()
    
  def train_and_evaluate( self, network_spec, nepoch=1, timeout=None, test=False, gpu_logger=None ):
    """ Train the specified network and return its error.
    
    Parameters:
      `network_spec` : Network specification
      `nepoch` : Number of training epochs
      `timeout` : Timeout for worker process
      `test` : Evaluate on test set in addition to validation set
    """
    f = self._pool.submit( Pool._train_and_evaluate,
      self._step, self._storage, self._backend, self._parameters, network_spec,
      nepoch, self._gpu_assignments, test=test, gpu_logger=gpu_logger )
    self._step += 1
    return f
