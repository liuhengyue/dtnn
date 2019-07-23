import logging
import os
import os.path

class ControllerBase:
  """ The base class for training controllers. Various callbacks allow the
  controller to monitor and control the training process.
  """

  def begin_epoch( self, model ):
    """ Called before a training epoch begins.
    
    Returns:
      True if training should proceed, False to stop training.
    """
    return False
    
  def finish_epoch( self, model ):
    """ Called after a training epoch is completed.
    """
    pass
    
  def begin_batch( self, model, batch ):
    """ Called before a training batch begins.
    """
    pass
  
  def finish_batch( self, model, loss ):
    """ Called after a training batch is completed.
    """
    pass
    
  def begin_training( self, model ):
    """ Called in each epoch before training begins.
    """
    pass
    
  def finish_training( self, model ):
    """ Called in each epoch after training is completed.
    """
    pass
    
  def begin_validation( self, model ):
    """ Called in each epoch before evaluating the model on the validation set.
    
    Returns:
      If False, do not evaluate on the validation set in this epoch.
    """
    return False
    
  def finish_validation( self, model, error ):
    """ Called in each epoch after validation is completed.
    """
    pass
    
  def begin_testing( self, model ):
    """ Called in each epoch before evaluating the model on the test set.
    
    Returns:
      If False, do not evaluate on the test set in this epoch.
    """
    return False
    
  def finish_testing( self, model, error ):
    """ Called in each epoch after testing is completed.
    """
    pass
    
# ----------------------------------------------------------------------------

class TestController(ControllerBase):
  """ Basic training controller implementation. Logs training steps and
  periodically saves the trained models.
  """
  def __init__( self, model, nepoch=1, outdir=os.getcwd(), model_save_interval=1 ):
    self._log = logging.getLogger( "nnsearch.backend.architect.TestController" )
    self._epoch = 0
    self._nepoch = nepoch
    self._outdir = outdir
    self._model_save_interval = model_save_interval

  def begin_epoch( self, model ):
    self._log.info( "begin_epoch(): epoch={};args={}".format( self._epoch, model.parameters() ) )
    return self._epoch < self._nepoch
    
  def finish_epoch( self, model ):
    self._log.info( "finish_epoch(): epoch={};args={}".format( self._epoch, model.parameters() ) )
    self._epoch += 1
    # Always save the last model
    if self._epoch == self._nepoch or self._epoch % self._model_save_interval == 0:
      filename = os.path.join( self._outdir, "model_{}.pkl.gz".format( self._epoch - 1 ) )
      model.save_model( filename )
    
  def begin_batch( self, model, batch ):
    self._log.info( "begin_batch(): epoch={};batch={}".format( self._epoch, batch ) )
  
  def finish_batch( self, model, loss ):
    self._log.info( "finish_batch(): epoch={};loss={}".format( self._epoch, loss ) )
    
  def begin_training( self, model ):
    self._log.info( "begin_training()" )
    
  def finish_training( self, model ):
    self._log.info( "finish_training()" )
    
  def begin_validation( self, model ):
    self._log.info( "begin_validation()" )
    return True
    
  def finish_validation( self, model, error ):
    self._log.info( "finish_validation(): error={}".format( error ) )
    
  def begin_testing( self, model ):
    self._log.info( "begin_testing()" )
    return True
    
  def finish_testing( self, model, error ):
    self._log.info( "finish_testing(): error={}".format( error ) )
