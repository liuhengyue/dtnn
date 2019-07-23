import gzip
import logging
import os
import sys
import time

from six.moves import cPickle

import numpy
import theano
import theano.misc.pkl_utils
import theano.tensor as T

import nnsearch.feedforward.Architect as ffarch
import nnsearch.feedforward.simple_load_data as ffld

class Instrumented:
  """ A wrapper for Architect that supports additional instrumentation.
  """
  def __init__( self, args ):
    self._args = args
    if args.load:
      (self._arch, self._args) = Instrumented.load_model( self._args.load, self._args )
    else:
      self._arch = ffarch.Architect(self._args)
    self._loader = ffld.SimpleLoadData(self._args)
    
  @staticmethod
  def load_model(filename, args):
    print('Loading model from filename {}'.format(filename))
    
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
    return (new_obj, new_args)
  
  def save_model( self, arch, args, epoch ):
    filename = os.sep.join( [args.output, "model_{}.pkl.gz".format( epoch )] )
    write_obj = {'args': args, 'mode': 'high_precision'}
    write_obj.update({lyr : [p.get_value() for p in arch.layers[lyr].param] for lyr in arch.layer_alias if arch.layers[lyr].param})
    
    with gzip.open(filename, 'wb') as gf:
      cPickle.dump(write_obj, gf) 
  
  def dump_parameters( self, epoch ):
    model_file = os.sep.join( [args.output, "model_" + str(epoch) + ".zip"] )
    with open( model_file, "wb" ) as fout:
      theano.misc.pkl_utils.dump( (self._arch.W_params), fout )
      
  def train( self ):
    loader = self._loader
    arch = self._arch
    
    train = arch.get_train_fn(loader.training)
    validate = arch.get_evaluator_fn(loader.validation)
    test = arch.get_evaluator_fn(loader.testing)
    #validation_output = arch.get_outputs_fn(loader.validation)
    
    # FIXME: 'args' is being pulled from outside the class scope! It's probably
    # the one instantiated in __main__
    train_nbatches = int(loader.training.numpy_x.shape[0]/args.batch)
    valid_nbatches = int(loader.validation.numpy_x.shape[0]/args.batch)
    test_nbatches = int(loader.testing.numpy_x.shape[0]/args.batch)
    
    cur_epoch = 0
    cur_batch = 0
    
    def train_instrument( xs, ys ):
      return theano.function( xs, ys, givens={
        arch.batch_input  : loader.training.gpu_x[arch.gpu_batch_idx], 
        arch.batch_labels : loader.training.gpu_y[arch.gpu_batch_idx]} )
    
    # Functions for data collection
    grad_magnitude = train_instrument([arch.gpu_batch_idx], [T.mean(abs(wg)) for wg in arch.W_grads])
    grad_norm2 = train_instrument([arch.gpu_batch_idx], [wg.norm(2) for wg in arch.W_grads])
    grad_std = train_instrument([arch.gpu_batch_idx], [T.std(wg) for wg in arch.W_grads])
    
    logger = logging.getLogger( "nnsearch.instrument" )
    # Names of layers that have learnable parameters
    logger.info( "W_param={}".format(arch.W_params))
    
    for batch in loader.training.batches():
      if cur_batch == 0:
        logger.info( "epoch={};args={}".format( cur_epoch, args ) )
      
      def my_train_fn( batch ):
        res = train( batch )
        gmagnitude = grad_magnitude( batch )
        gnorm2 = grad_norm2( batch )
        gstd = grad_std( batch )
        logger.info( "epoch={};batch={};grad_magnitude={};grad_norm2={};grad_std={}".format(
                     cur_epoch, cur_batch, gmagnitude, gnorm2, gstd) )
        return res
      
      res = my_train_fn( batch )
      logger.info("epoch={};batch={};loss={}".format(cur_epoch, cur_batch, res))
      
      if cur_batch == train_nbatches - 1:
        # Training finished
        if cur_epoch == (args.nepoch - 1) or cur_epoch % self._args.model_save_interval == 0:
          # self.dump_parameters( cur_epoch )
          self.save_model( arch, args, cur_epoch )
        
        for phase in ["valid", "test"]:
          error = self.evaluate( phase )
          logger.info( "epoch={};phase={};error={}".format( cur_epoch, phase, error ) )
        
        args.learning_rate = args.decay * args.learning_rate
        cur_batch, cur_epoch = 0, cur_epoch + 1
      else:
        cur_batch, cur_epoch = cur_batch + 1, cur_epoch
          
      if cur_epoch == args.nepoch:
        break
  
  def evaluate( self, phase ):
    loader = self._loader
    arch = self._arch
    args = self._args
    
    if phase == "valid":
      src = loader.validation
    elif phase == "test":
      src = loader.testing
    else:
      raise ValueError( "phase" )
      
    eval_fn = arch.get_evaluator_fn( src )
    nbatches = int(src.numpy_x.shape[0]/args.batch)
    
    errs = []
    for batch in src.batches():
      errs.append( eval_fn( batch ) )
      if len(errs) == nbatches:
        break
    error = numpy.mean( errs )
    return error
    
  def test( self ):
    logger = logging.getLogger( "nnsearch.instrument" )
    error = self.evaluate( "test" )
    logger.info( "test;error={}".format( error ) )
  
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
  assert( args.test == no_arch )
  
  instrumented = Instrumented( args )
  if args.test:
    instrumented.test()
  else:
    instrumented.train()
