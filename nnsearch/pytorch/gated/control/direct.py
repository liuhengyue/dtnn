import abc
import itertools
import logging
import math

import numpy as np

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fn
from   torch.nn.parameter import Parameter

import nnsearch.pytorch.gated.strategy as strategy
from   nnsearch.pytorch.modules import FullyConnected
import nnsearch.pytorch.torchx as torchx

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

class BlindController(nn.Module, strategy.GatePolicy):
  def __init__( self, control, output ):
    super().__init__()
    self.control = control
    self.output = output
    
  def reset( self, x ):
    pass
  
  def next_module( self, m ):
    raise NotImplementedError()
    
  def set_control( self, u ):
    self._u = u
    
  def forward( self, x ):
    g = self.control( self._u.unsqueeze(-1) )
    g = self.output( g )
    return g

# ----------------------------------------------------------------------------

class IndependentController(nn.Module, strategy.GatePolicy):
  def __init__( self, input, control, output ):
    super().__init__()
    self.input = input
    self.control = control
    self.output = output
    
  def reset( self, x ):
    pass
    
  def next_module( self, m ):
    raise NotImplementedError()
    
  def set_control( self, u ):
    self._u = u
    
  def forward( self, x ):
    log.debug( "direct.independent: x: %s", x.shape )
    # x = self.input( x.detach() )
    x = self.input( x )
    x = torch.cat( [self._u.unsqueeze(-1), x], dim=1 )
    x = self.control( x )
    x = self.output( x )
    return x

# ----------------------------------------------------------------------------

class RecurrentController(nn.Module, strategy.GatePolicy):
  """ Makes gate decisions sequentially based on the output of the previous
  layer, the control signal, and recurrent state.
  
  Parameters:
    `recurrent_cell` : Module implementing one step of an RNN
    `hidden_size` : Size of RNN hidden state
    `input` : If not `None`
    `output` : If not `None`, a module applied to the RNN output
    `use_cuda` : Move module parameters to GPU
  """
  def __init__( self, recurrent_cell, hidden_size, input=None,
                output=None, use_cuda=False ):
    super().__init__()
    self.rnn = recurrent_cell
    if input is None:
      self.input = None
    elif isinstance(input, nn.Module):
      self.input = input
    else:
      self.input = nn.ModuleList( input )
    self.output = output
    h0 = torch.randn( 1, hidden_size )
    c0 = torch.randn( 1, hidden_size )
    if use_cuda:
      h0 = h0.cuda()
      c0 = c0.cuda()
    self.h0 = Parameter(h0)
    self.c0 = Parameter(c0)
    self._i = None
    self._u = None
    
  def reset( self ):
    self._i = -1
    self._h = None
    self._c = None
    
  def next_module( self, m ):
    self._i += 1
  
  def set_control( self, u ):
    assert( (u >= 0.0).all() )
    assert( (u <= 1.0).all() )
    self._u = u
    
  def _input( self, x ):
    if self.input is None:
      return x
    elif isinstance(self.input, nn.ModuleList):
      return self.input[self._i]( x )
    else:
      return self.input( x )
  
  def forward( self, x ):
    log.debug( "gate.forward: layer: %s; control: %s", self._i, self._u )
    batch_size = x.size(0)
    # Empty state
    if self._i == 0:
      self._h = self.h0.repeat( batch_size, 1 )
      self._c = self.c0.repeat( batch_size, 1 )
    x = self._input( x.detach() )
    x = torch.cat( [self._u.unsqueeze(-1), x], dim=1 )
    log.debug( "gate.forward: [u, x]: %s", x )
    self._h, self._c = self.rnn( x, (self._h, self._c) )
    if self.output is not None:
      return self.output( self._h )
    else:
      return self._h
