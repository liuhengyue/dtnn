import argparse
import contextlib
import glob
import logging
import os
import re
import shlex
import shutil
import sys
import nnsearch.logging as mylog
import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.init as init
from   torch.nn.parameter import Parameter
import torch.optim as optim

log = logging.getLogger( __name__ )
mylog.add_log_level("VERBOSE", logging.INFO - 5)
# ----------------------------------------------------------------------------

class CheckpointManager:
  def __init__( self, *, output, input=None ):
    self.output = output
    self.input  = input

  def model_file( self, directory, prefix, epoch, suffix="" ):
    filename = "{}_{}.pkl{}".format( prefix, epoch, suffix )
    return os.path.join( directory, filename )
    
  def latest_checkpoints( self, directory, name ):
    return glob.glob( os.path.join(directory, "{}_*.pkl.latest".format(name)) )
    
  def epoch_of_model_file( self, path ):
    m = re.match( ".*_([0-9]+)\\.pkl(\\.latest)?", os.path.basename(path) ).group(1)
    return int(m)

  # --------------------------------------------------------------------------
  # Saving

  def save_checkpoint( self, name, network, elapsed_epochs, *,
                       data_parallel, persist=False ):
    # Save current model to tmp name
    tokens = (self.output, name, elapsed_epochs)
    with open( self.model_file(*tokens, ".tmp"), "wb" ) as fout:
      if data_parallel:
        # See: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/19
        torch.save( network.module.state_dict(), fout )
      else:
        torch.save( network.state_dict(), fout )
    # Remove previous ".latest" checkpoints
    for f in self.latest_checkpoints( self.output, name ):
      with contextlib.suppress(FileNotFoundError):
        os.remove( f )
    # Move tmp file to latest
    os.rename( self.model_file(*tokens, ".tmp"), 
               self.model_file(*tokens, ".latest") )
    if persist:
      shutil.copy2( self.model_file(*tokens, ".latest"),
                    self.model_file(*tokens) )
    self.on_save_finished()
    
  def on_save_finished( self ):
    pass
    
  # --------------------------------------------------------------------------
  # Loading
  
  def load_checkpoint( self, name, network, checkpoint, strict=True, skip=None ):
    path = self.get_checkpoint_file( name, checkpoint )
    self.load_parameters( path, network, strict=strict, skip=skip )
  
  def get_checkpoint_file( self, name, checkpoint ):
    if checkpoint == "latest":
      latest = self.latest_checkpoints( self.input, name )
      if len(latest) > 1:
        raise RuntimeError( "multiple .latest model files" )
      filename = latest[0]
    else:
      try:
        load_epoch = int(checkpoint)
      except (TypeError, ValueError):
        raise ValueError( "'checkpoint' must be \"latest\" or integer epoch" )
      filename = self.model_file( self.input, name, load_epoch )
    return filename
  
  def load_parameters( self, path, network, strict=True, skip=None, map_location=None ):
    if skip is None:
      skip = lambda param_name: False
      
    with open( path, "rb" ) as fin:
      state_dict = torch.load( fin, map_location="cpu" )

    own_state = network.state_dict()

    def load_layer_by_name(name, param):
      log.verbose("Load %s", name)

      if isinstance(param, Parameter):
        # backwards compatibility for serialized parameters
        param = param.data
      try:
        own_state[name].copy_(param)
      except Exception:
        raise RuntimeError('While copying the parameter named {}, '
                           'whose dimensions in the model are {} and '
                           'whose dimensions in the checkpoint are {}.'
                           .format(name, own_state[name].size(), param.size()))

    for name, param in state_dict.items():
      if name in own_state:
        if skip(name):
          log.verbose("Skipping module")
          continue
        load_layer_by_name(name, param)
      elif strict:
        raise KeyError( "unexpected key '{}' in state_dict".format(name) )
      elif "module." in name:
        new_name = name.replace("module.", "")
        load_layer_by_name(new_name, param)
      else:
        log.warning( "unexpected key '{}' in state_dict".format(name) )
    
    missing = set(own_state.keys()) - set(state_dict.keys())
    missing = [k for k in missing if not skip(k)]
    if len(missing) > 0:
      if strict:
        raise KeyError( "missing keys in state_dict: {}".format(missing) )
      else:
        log.warning( "missing keys in state_dict: {}".format(missing) )
    log.info("Loaded model from %s", path)
