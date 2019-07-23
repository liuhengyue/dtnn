import sys

sys.path.append('/home/local/SRI/e29288/AC/nnsearch/nnsearch/faster-rcnn/model')
sys.path.append('/home/local/SRI/e29288/AC/nnsearch/nnsearch/faster-rcnn/utils')


sys.path.append('../model')
print(sys.path)
from torch import nn
from torchvision.models import densenet121, densenet169 
import torch as t 
from torch.autograd import Variable
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt
import ipdb


#TODO: Implement
def gated_densenet_backbone():
    pass


def densenet_backbone(opt):
    if opt.torch_pretrain:
        model = densenet169(pretrained=True)
    elif not opt.model_pretrain_path:
        model = densenet169(pretrained=False)
        model.load_state_dict(t.load(opt.model_pretrain_path))
    else:
        logging.info('The path to the pretrained model file not specified. Loading default torch pretrained file')
        model = densenet169(not opt.model_pretrain_path)


    features = list(model.features)[:9] ## removed last two as they are added in the classsifier
    # add the last two features to classifier
    classifier  = list(model.features)[9:]
    # add a global averaging layer
    classifier.append(nn.AvgPool2d((7,7)))

    #### freeze top dense layers
    for layer in features[:7]:
        for p in layer.parameters():
            p.requires_grad=False

    return nn.Sequential(*features), nn.Sequential(*classifier)





class FasterRCNNDensenet(FasterRCNN):
    """Faster R-CNN based on Densenet.
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

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 opt,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], gated=False
                 ):
                 
        if gated:
            extracter, classifier = gated_densenet_backbone()
        else:
            extractor, classifier = densenet_backbone(opt)
        rpn = RegionProposalNetwork(
            1280,1280,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = DensenetRoIHead(
            n_class=n_fg_class + 1,
            roi_size=14,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNDensenet, self).__init__(
            extractor,
            rpn,
            head,
        )


class DensenetRoIHead(nn.Module):
    """Faster R-CNN Head for Densenet based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,classifier
                 ):
        # n_class includes the background
        super(DensenetRoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(1664, n_class * 4)
        self.score = nn.Linear(1664, n_class)

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
        #pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool.squeeze().squeeze()) # KA added squeeze
        roi_cls_locs = self.cls_loc(fc7.squeeze().squeeze())
        roi_scores = self.score(fc7.squeeze().squeeze())
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

