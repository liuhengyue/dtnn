import torch as t
from   torch import nn
from torchvision.models import vgg16
from nnsearch.faster_rcnn.model.region_proposal_network import RegionProposalNetwork
from nnsearch.faster_rcnn.model.faster_rcnn import FasterRCNN
from nnsearch.faster_rcnn.model.roi_module import RoIPooling2D
from nnsearch.faster_rcnn.utils import array_tool as at
from nnsearch.faster_rcnn.utils.config import opt

from   nnsearch.pytorch import torchx
import nnsearch.pytorch.gated.densenet as dn
import nnsearch.pytorch.gated.module as gmod
import nnsearch.pytorch.gated.strategy as strategy
from   nnsearch.pytorch.modules import (
  FrozenBatchNorm, FullyConnected, GlobalAvgPool2d)


def _decompose_densenet169bc( network ):
  n = 21 # Length of network.fn
  # Only works with simple gate controllers
  assert isinstance(network.gate, strategy.SequentialGate)
  # network.fn is a ModuleList with 21 elements
  assert isinstance(network.fn, nn.ModuleList)
  assert len(network.fn) == n
  # Layer 15 is the AvgPool2d layer that we will replace with RoIPooling
  assert isinstance(network.fn[15], nn.AvgPool2d)
  
  # Split the data path; exclude layer 15 AvgPool2d
  fn_layers = [m for m in network.fn] # ModuleList doesn't support slicing
  tail_start = 12 # Split between 3rd DenseNet block and 3rd Transition
  head_fn_layers = fn_layers[:tail_start]
  tail_fn_layers = fn_layers[tail_start:n-1] # Drop final FC layer
  assert isinstance(head_fn_layers[-1], dn.GatedDenseNetBlock)
  assert isinstance(tail_fn_layers[0], nn.BatchNorm2d)
  assert isinstance(tail_fn_layers[-1], GlobalAvgPool2d)
  
  # Split the gate network
  # SequentialGate is just a list of gate controllers
  gate_layers = list(network.gate.gate_modules) # ModuleList doesn't support slicing
  head_gate_layers = gate_layers[:3]
  tail_gate_layers = gate_layers[3:]
  
  # Only fine-tune the final (Transition->DenseBlock) of `head`
  for m in head_fn_layers[:-5]:
    torchx.freeze( m )
  
  def construct( fn_layers, gate_layers ):
    gated_modules = [m for m in fn_layers if isinstance(m, dn.GatedDenseNetBlock)]
    gate = strategy.SequentialGate( gate_layers )
    gnet = gmod.GatedChainNetwork( gate, fn_layers, gated_modules )
    # Freeze batch norm layers (in data path only) as in [He et al., 2016]
    gnet.fn = FrozenBatchNorm( gnet.fn )
    return gnet
  
  head = construct( head_fn_layers, head_gate_layers )
  tail = construct( tail_fn_layers, tail_gate_layers )

  return (head, tail)

# TODO: The only thing specific to DenseNet-169BC is the number of features
# for the RPN, which could easily be deduced from the network instance.
class FasterRCNNDenseNet169BC(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    def __init__(self, network,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
      extractor, classifier = _decompose_densenet169bc( network )
      extractor = torchx.IndexSelect(extractor, 0)
      classifier = torchx.IndexSelect(classifier, 0)

      rpn = RegionProposalNetwork(
          1280, 1280,
          ratios=ratios,
          anchor_scales=anchor_scales,
          feat_stride=16,
      )

      head = RoIHeadDenseNet(
          n_class=n_fg_class + 1,
          classifier=classifier,
          classifier_channels=1664
      )

      super().__init__( extractor, rpn, head )


class RoIHeadDenseNet(nn.Module):
    """Faster R-CNN Head for DenseNet-based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        classifier (nn.Module): The classifier portion of the network
        classifier_channels (int): Number of output channels of classifier.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
    """

    def __init__( self, n_class, classifier, classifier_channels,
                  roi_size=14, spatial_scale=(1 / 16) ):
        # n_class includes the background
        super().__init__()

        self.classifier = classifier
        self.cls_loc = FullyConnected(classifier_channels, n_class * 4)
        self.score = FullyConnected(classifier_channels, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

        pool = self.roi(x, indices_and_rois)
        classifier_features = self.classifier(pool)
        roi_cls_locs = self.cls_loc(classifier_features)
        roi_scores = self.score(classifier_features)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
