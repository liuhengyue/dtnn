import gzip
import logging
import os
import sys
import time

from six.moves import cPickle

import numpy

from nnsearch.backend.training import ControllerBase, TestController
from nnsearch.storage import ModelDirectory

class FeedForward:
  """ Standardized interface to feedforward.Architect.
  """
  
  @staticmethod
  def gpu_setup( gpu_id, gpu_name=None ):
    log = logging.getLogger( __name__ )
    gpu_str = "gpu" + str(gpu_id)
    if gpu_name:
      os.environ["THEANO_FLAGS"] += ",compiledir_format=compiledir_" + gpu_name + "-%(processor)s-%(python_version)s-%(python_bitwidth)s"
    import theano.sandbox.cuda
    theano.sandbox.cuda.use( gpu_str )
    import theano
    log.info( theano.config )
  
  def __init__( self, arch, args ):
    import nnsearch.feedforward.simple_load_data as ffld
  
    self.logger = logging.getLogger( __name__ )
    self.logger.info( "creating FeedForward with arguments: %s", args )
  
    self._arch = arch
    self._args = args
    
    self._loader = ffld.SimpleLoadData(self._args)
    self._train = self._arch.get_train_fn(self._loader.training)
    self._validate = self._arch.get_evaluator_fn(self._loader.validation)
    self._test = self._arch.get_evaluator_fn(self._loader.testing)
    
  def parameters( self ):
    return self._args
  
  @staticmethod
  def create_model( args, s ):
    import nnsearch.feedforward.Architect as ffarch
    # TODO: Move string encoding logic to 'backend' implementation
    args.arch = s.arch_string()
    arch = ffarch.Architect( args )
    return FeedForward( arch, args )
  
  @staticmethod
  def load_model(filename, args):
    import nnsearch.feedforward.Architect as ffarch
    
    logging.getLogger( __name__ ).info( "loading %s", filename )
    
    with gzip.open(filename, 'rb') as gf:
      read_obj = cPickle.load(gf)
      assert read_obj['mode'] == 'high_precision'
      new_args = read_obj["args"]
      new_args.output = args.output
      new_obj = ffarch.Architect(new_args)
      
      for lyr in new_obj.layer_alias:
        if lyr in read_obj:
          for idx_p, p in enumerate(read_obj[lyr]):
            new_obj.layers[lyr].param[idx_p].set_value(p)
    return FeedForward( new_obj, new_args )
  
  def save_model( self, filename ):
    write_obj = {'args': self._args, 'mode': 'high_precision'}
    write_obj.update({lyr : [p.get_value() for p in self._arch.layers[lyr].param] 
                      for lyr in self._arch.layer_alias if self._arch.layers[lyr].param})
    
    with gzip.open(filename, 'wb') as gf:
      cPickle.dump(write_obj, gf) 
      
  def train( self, controller ):
    import theano
    import theano.tensor as T
  
    train_nbatches = int(self._loader.training.numpy_x.shape[0]/self._args.batch)
    valid_nbatches = int(self._loader.validation.numpy_x.shape[0]/self._args.batch)
    test_nbatches = int(self._loader.testing.numpy_x.shape[0]/self._args.batch)
    
    cur_epoch = 0
    cur_batch = 0
    
    def train_instrument( xs, ys ):
      return theano.function( xs, ys, givens={
        self._arch.batch_input  : self._loader.training.gpu_x[self._arch.gpu_batch_idx], 
        self._arch.batch_labels : self._loader.training.gpu_y[self._arch.gpu_batch_idx]} )
    
    # Functions for data collection
    grad_magnitude = train_instrument([self._arch.gpu_batch_idx], [T.mean(abs(wg)) for wg in self._arch.W_grads])
    grad_norm2 = train_instrument([self._arch.gpu_batch_idx], [wg.norm(2) for wg in self._arch.W_grads])
    grad_std = train_instrument([self._arch.gpu_batch_idx], [T.std(wg) for wg in self._arch.W_grads])
    
    # Names of layers that have learnable parameters
    self.logger.info( "W_param={}".format(self._arch.W_params))
    
    if controller.begin_epoch( self ):
      controller.begin_training( self )
      
      for batch in self._loader.training.batches():
        controller.begin_batch( self, cur_batch )
        
        def my_train_fn( batch ):
          loss = self._train( batch )
          gmagnitude = grad_magnitude( batch )
          gnorm2 = grad_norm2( batch )
          gstd = grad_std( batch )
          self.logger.info( "epoch={};batch={};grad_magnitude={};grad_norm2={};grad_std={}".format(
                       cur_epoch, cur_batch, gmagnitude, gnorm2, gstd) )
          return loss
        
        loss = my_train_fn( batch )
        controller.finish_batch( self, loss )
        
        if cur_batch == train_nbatches - 1:
          # Training finished
          controller.finish_training( self )
          
          if controller.begin_validation( self ):
            error = self.validate()
            controller.finish_validation( self, error )
          
          if controller.begin_testing( self ):
            error = self.test()
            controller.finish_testing( self, error )
            
          controller.finish_epoch( self )
          
          self._args.learning_rate = self._args.decay * self._args.learning_rate
          cur_batch, cur_epoch = 0, cur_epoch + 1
          
          if controller.begin_epoch( self ):
            controller.begin_training( self )
          else:
            break
        else:
          cur_batch, cur_epoch = cur_batch + 1, cur_epoch
  
  def _evaluate( self, phase ):
    if phase == "validation":
      src = self._loader.validation
      eval_fn = self._validate
    elif phase == "testing":
      src = self._loader.testing
      eval_fn = self._test
    else:
      raise ValueError( "phase" )
      
    nbatches = int(src.numpy_x.shape[0]/self._args.batch)
    errs = []
    for batch in src.batches():
      errs.append( eval_fn( batch ) )
      if len(errs) == nbatches:
        break
    error = numpy.mean( errs )
    return error
  
  def validate( self ):
    error = self._evaluate( "validation" )
    return error
  
  def test( self ):
    error = self._evaluate( "testing" )
    return error
    
if __name__ == "__main__":
  parser = ffarch.get_parser()
  parser.add_argument( "-model_save_interval", type=int, default=1, help="Save the model when (epoch % interval) == 0." )
  parser.add_argument( "-load", type=str, default=None, help="Load model from file." )
  parser.add_argument( "-test", action="store_true", help="Test only" )
  argv = sys.argv[1:]
  no_arch = False
  if argv[1].startswith( "-" ):
    # Second positional argument not supplied -> insert dummy
    argv.insert( 1, "None" )
    no_arch = True
  args = parser.parse_args( argv )
  # assert( args.test == no_arch )
  
  class Train(ControllerBase):
    def __init__( self, nepoch = 1 ):
      self._nepoch = nepoch
      self._epoch = 0
      
    def begin_epoch( self, model ):
      return self._epoch < self._nepoch
    
    def finish_epoch( self, model ):
      self._epoch += 1
  
  backend = FeedForward
  storage = ModelDirectory( args.output, ".tgz" )
  if args.load:
    model = backend.load_model( storage.get_filename( args.arch ), args )
  else:
    model = backend.create_model( args, args.arch )
  if args.test:
    error = model.test()
    print( "Test error: {}".format( error ) )
  else:
    # controller = TestController( model, nepoch=args.nepoch, model_save_interval=args.model_save_interval )
    controller = Train( nepoch = 1 )
    model.train( controller )
    model.save_model( storage.get_filename( args.arch ) )
