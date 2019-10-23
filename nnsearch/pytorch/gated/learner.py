import abc
import contextlib
import logging

import torch
from   torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import nnsearch.pytorch.parameter as parameter
import nnsearch.pytorch.torchx as torchx
from   nnsearch.statistics import MeanAccumulator

log = logging.getLogger( __name__ )

# ----------------------------------------------------------------------------

# FIXME: This class is not specific to gated networks
class Learner:
  def __init__( self, network, optimizer, learning_rate, criterion=None):
    self.network = network
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.criterion = nn.CrossEntropyLoss( reduction='none' ) if criterion == None else criterion
    
    self.histogram_bins = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    self.grad_histogram = None
  
  def _class_loss( self, yhat, labels ):
    # Classification error
    batch_size = yhat.size(0)
    loss = self.criterion( yhat, labels )
    if isinstance(self.criterion, nn.CrossEntropyLoss):
      _, predicted = torch.max( yhat.data, 1 )
      correct = torch.sum( predicted == labels.data )
      log.debug( "loss.errors: %s", batch_size - correct )
    
    return loss
  
  def loss( self, yhat, labels ):
    """ Returns a vector of length B containing the loss for each batch element.
    """
    return self._class_loss( yhat, labels )
    
  def start_train( self, epoch_idx, seed ):
    self.network.train()
    
    if log.isEnabledFor( logging.VERBOSE ):
      self.grad_histogram = [(b, 0) for b in self.histogram_bins]
    
    # Update learning rate
    # FIXME: Should we do this in start_batch() for maximum Cosine?
    for param_group in self.optimizer.param_groups:
      #param_group["lr"] = self.learning_rate()
      param_group["lr"] = self.learning_rate
    
    self.running_loss = MeanAccumulator()
    log.info( "learner: Epoch %s: seed=%s; learning_rate=%s",
              epoch_idx, seed, self.learning_rate )
    
  def start_eval( self, epoch_idx, seed ):
    self.network.eval()
    
  def finish_train( self, epoch_idx ):
    if log.isEnabledFor( logging.VERBOSE ):
      log.verbose( "learner.grad_histogram:\n%s",
        torchx.pretty_histogram( self.grad_histogram ) )
      
      whist = [(b, 0) for b in self.histogram_bins]
      for p in torchx.optimizer_params( self.optimizer ):
        h = torchx.histogram( torch.abs(p.data), self.histogram_bins )
        whist = [(b, x+y) for ((b, x), (_, y)) in zip(whist, h)]
      log.verbose( "learner.w_mag_histogram:\n%s",
        torchx.pretty_histogram( whist ) )
  
    log.info( "train {:d} loss: {:.3f}".format(
      epoch_idx, self.running_loss.mean() ) )
      
  def finish_eval( self, epoch_idx ):
    pass
    
  def start_batch( self, batch_idx, inputs, labels ):
    pass
  
  def measure( self, batch_idx, inputs, labels, yhat ):
    pass
  
  def forward( self, batch_idx, inputs, labels ):
    self.start_batch( batch_idx, inputs, labels )
    
    with contextlib.ExitStack() as ctx:
      if not self.network.training:
        ctx.enter_context( torch.no_grad() )
      result = self._network_forward( inputs ) # removed variable
      if isinstance(result, tuple):
        yhat, *self.rest = result
      else:
        yhat = result
        self.rest = []
      log.debug( "learner.forward: yhat: %s, rest: %s", yhat, self.rest )
      return yhat
      
  def _network_forward( self, inputs_var ):
    return self.network( inputs_var )
    
  def backward( self, batch_idx, yhat, labels ):
    # Zero entire network; FIXME: not necessary (wastes computation) but I
    # don't want to get burned later because I didn't do it.
    self.network.zero_grad()
    
    loss = self.loss( yhat, labels)
    loss = torch.mean( loss )
    log.debug( "learner.loss: %s", loss.item() )
    self.running_loss( loss.item() )
    
    # Optimization
    loss.backward()
    
    if log.isEnabledFor( logging.MICRO ):
      log.micro( "Gradients" )
      for (name, p) in self.network.named_parameters():
        log.micro( "grad: %s: %s", name, p.grad )
    
    if log.isEnabledFor( logging.VERBOSE ):
      for p in torchx.optimizer_params( self.optimizer ):
        if p.grad is not None:
          h = torchx.histogram( torch.abs(p.grad.data), self.histogram_bins )
          self.grad_histogram = [
            (b, x+y) for ((b, x), (_, y)) in zip(self.grad_histogram, h)]
    self.optimizer.step()
    self.rest = None
    return loss
# ----------------------------------------------------------------------------

class GatedNetworkLearner(Learner):
  def __init__( self, network, optimizer, learning_rate,
      gate_policy, gate_control, **kwargs ):
    super().__init__( network, optimizer, learning_rate, **kwargs )
    self.gate_policy = gate_policy
    self.gate_control = gate_control

  def start_batch( self, batch_idx, inputs, labels ):
    super().start_batch( batch_idx, inputs, labels )
    log.debug( "input.size: %s", inputs.size() )
    self.u = self.gate_control( inputs, labels )
    log.debug( "batch.u: %s", self.u )
      
  def _network_forward( self, inputs_var ):
    return self.network( inputs_var, self.u )
  
  def measure( self, batch_idx, inputs, labels, yhat ):
    super().measure( batch_idx, inputs, labels, yhat )
    _, predicted = torch.max( yhat, 1 )
    Gs = [torch.mean((g > 0).type_as(yhat), dim=1) for (g, info) in self.rest[0]]
    head = "|".join( "G{}".format(i) for i in range(len(Gs)) )
    G = torch.stack( Gs, dim=1 )
    inputs = torch.stack(
      [self.u.data, labels.type_as(G.data), predicted.type_as(G.data)], dim=1 )
    data = torch.cat( [G.data, inputs], dim=1 )
    with torchx.printoptions_nowrap( profile="full" ):
      log.info( "%s|u|labels|predicted:\n%s", head, data )
    
# ----------------------------------------------------------------------------
    
class GatedDataPathLearner(GatedNetworkLearner):
  """ Learner for training the entire network. Use with a fixed gating
  strategy for two-phase gated network training.

  """
  def __init__( self, network, optimizer, learning_rate, gate_policy, gate_control,
                scheduler=None, **kwargs ):
    super().__init__( network, optimizer, learning_rate, gate_policy, gate_control, **kwargs )
    self.scheduler = scheduler


  def loss( self, yhat, labels ):
    return self._class_loss( yhat, labels )

  def scheduler_step(self, loss, epoch):
    self.scheduler.step(loss, epoch)

  def update_gate_control(self, gate_control, u_stage=None):
    self.gate_control = gate_control
    # give the u_stage, freeze the first c components
    if u_stage:
      u_stage_l, u_stage_r = u_stage
      gated_modules = self.network.module._gated_modules \
        if isinstance(self.network, torch.nn.DataParallel) else self.network._gated_modules
      for gated_module in gated_modules:
          # set all to false
          for component in gated_module[0].components:
            for p in component.parameters():
              p.requires_grad = False

      for gated_module in gated_modules:
          n = len(gated_module[0].components)
          # set active components to True
          c_l = int(n * u_stage_l)
          c_r = int(n * u_stage_r)
          c_r = min(c_r + 1, n) if c_r == c_l else c_r
          assert c_r > c_l, "Error: No active components."
          for i in range(c_l, c_r):
            for p in gated_module[0].components[i].parameters():
              p.requires_grad = True
      # check the requires grad term
      # for gated_module in gated_modules:
      #     print(gated_module)
      #     for component in gated_module[0].components:
      #       for p in component.parameters():
      #         print(p.requires_grad, end=' ')


    
# ----------------------------------------------------------------------------

def usage_gate_loss( penalty_fn ):
  """ The mean loss from gated module usage, measured by `penalty_fn` and
  weighted by complexity. `info` is unused.
  """
  def f( gs, info, u, weights ):
    # Un-normalized binary gate matrices
    assert( len(gs) == len(weights) )
    G = sum( torch.mean(g * w, dim=1) for (g, w) in zip(gs, weights) )
    log.verbose( "adaptive.loss.G: %s", G )
    log.verbose( "adaptive.loss.u: %s", u )
    Lgate = penalty_fn( G, u )
    log.verbose( "loss.Lgate: %s", Lgate )
    Lg_bar = torch.mean( Lgate )
    log.info( "loss.Lg_bar: %s", Lg_bar.item() )
    return Lgate
  return f
  
def act_gate_loss( penalty_fn ):
  """ The ACT "ponder cost", weighted by complexity. `info` must be a list
  containing the `rho` values for each gated module.
  """
  def f( gs, info, u, weights ):
    rhos = info
    assert( len(gs) == len(rhos) == len(weights) )
    u = u.unsqueeze(-1)
    log.verbose( "Lgate.loss.u: %s", u )
    log.micro( "Lgate.act.rhos: %s", rhos )
    ncomps = [g.size(1) for g in gs]
    
    # ponder_costs = torch.cat( [rho.N + rho.R for rho in rhos], dim=1 )
    
    # Normalized count of active components
    Nbars = [rho.N / n for (rho, n) in zip(rhos, ncomps)]
    log.micro( "Lgate.act.Nbars: %s", Nbars )
    # Penalize if total weighted usage exceeds `u`
    usage = [Nbar * w for (Nbar, w) in zip(Nbars, weights)]
    usage = torch.cat( usage, dim=1 )
    log.micro( "Lgate.act.usage: %s", usage )
    sum_usage = torch.sum(usage, dim=1, keepdim=True)
    diff = sum_usage - u
    sign = torch.sign( diff )
    # penalty = sign * torch.pow(diff, 2)
    penalty = sign * penalty_fn( sum_usage, u )
    log.micro( "Lgate.act.penalty: %s", penalty )
    # penalize = (torch.sum( usage, dim=1, keepdim=True ) > u).type_as(u.data)
    
    # If count exceeds `u`, multiply it by the module's weight
    # ws = [(Nbar > u).type_as(u.data) * w for (Nbar, w) in zip(Nbars, weights)]
    # ws = torch.cat( ws, dim=1 )
    # log.debug( "Lgate.act.ws: %s", ws )
    
    rhos = torch.cat( [rho.N + rho.R for rho in rhos], dim=1 )
    # Ls = penalize * rhos
    Ls = penalty * rhos
    log.verbose( "Lgate.act.Ls: %s", Ls )
    L = torch.sum( Ls, dim=1 )
    return L
  return f
    
class GatePolicyLearner(GatedNetworkLearner):
  """ Basic learner for the gate policy. Combines the loss due to accuracy
  with the `gate_loss` weighted by `lambda_gate`.
  """
  def __init__( self, *args, gate_loss, lambda_gate, component_weights, **kwargs ):
    super().__init__( *args, **kwargs )
    self.gate_loss = gate_loss
    self.lambda_gate  = lambda_gate
    self.weights   = torch.Tensor( component_weights )
    self.weights /= torch.sum( self.weights )
    log.debug( "GatePolicyLearner.weights (normalized): %s", self.weights )
  
  def loss( self, yhat, labels ):
    Lacc = self._class_loss( yhat, labels )
    log.debug( "GatePolicyLearner.Lacc: %s", Lacc )
    
    gs, info = zip( *self.rest[0] )
    Lgate = self.gate_loss( gs, info, self.u, self.weights )
    log.debug( "GatePolicyLearner.Lgate: %s", Lgate )
    
    return Lacc + self.lambda_gate * Lgate
    
class ReinforceGateLearner(GatePolicyLearner):
  """ Uses the loss computed by the underlying `GatePolicyLearner` as the
  reward signal for the REINFORCE algorithm. The `info` object must be a list
  of gate policy probability vectors for each gated module.
  """
  def loss( self, yhat, labels ):
    # L = super().loss( yhat, labels )
    Lacc = self._class_loss( yhat, labels )
    log.debug( "GatePolicyLearner.Lacc: %s", Lacc )
    
    gs, info = zip( *self.rest[0] )
    Lgate = self.gate_loss( gs, info, self.u, self.weights )
    log.debug( "GatePolicyLearner.Lgate: %s", Lgate )
    
    # Reward
    Lg = Lacc + self.lambda_gate * Lgate
    R = -Lg.unsqueeze(-1)
    # R = -L.unsqueeze(-1)
    actions, probs = zip( *self.rest[0] )
    pgs = []
    log.debug( "reinforce.R: %s", R )
    for (g, p) in zip(actions, probs):
      log.debug( "reinforce.g: %s", g )
      log.debug( "reinforce.p: %s", p )
      # Action probabilities
      pa = g*p + (1.0 - g) * (1.0 - p)
      log.debug( "reinforce.pa: %s", pa )
      pg = R.detach() * torch.log( pa )
      log.debug( "reinforce.pg: %s", pg )
      pgs.append( pg )
    # The joint action probability is the product of probabilities for each
    # component -> reduce using sum.
    # Negate because we're minimizing.
    pg_loss = -torch.sum( sum( pgs ), dim=1 )
    # Note that the gate matrices are random samples and thus detached from
    # the computation graph. Therefore no gradients propagate to the PG learner
    # via the `Lacc` term
    return pg_loss + Lacc #self._class_loss( yhat, labels )

# FIXME: Incorporate the class loss gradient `Lacc` (see above)
class ReinforceCountGateLearner(GatePolicyLearner):
  """ Uses the loss computed by the underlying `GatePolicyLearner` as the
  reward signal for the REINFORCE algorithm. The `info` object must be a list
  of gate policy probability vectors for each gated module.
  """
  def loss( self, yhat, labels ):
    L = super().loss( yhat, labels )
    # Reward
    R = -L.unsqueeze(-1)
    actions, probs = zip( *self.rest[0] )
    pgs = []
    log.debug( "reinforce.R: %s", R )
    for (g, p) in zip(actions, probs):
      log.debug( "reinforce.g: %s", g )
      log.debug( "reinforce.p: %s", p )
      c = torch.sum( g, dim=1, keepdim=True ).long()
      log.debug( "reinforce.c: %s", c )
      # Action probabilities
      pa = torch.gather( p, 1, c )
      log.debug( "reinforce.pa: %s", pa )
      pg = R.detach() * torch.log( pa )
      log.debug( "reinforce.pg: %s", pg )
      pgs.append( pg )
    # The joint action probability is the product of probabilities for each
    # component -> reduce using sum.
    # Negate because we're minimizing.
    pg_loss = -torch.sum( sum( pgs ), dim=1 )
    return pg_loss
