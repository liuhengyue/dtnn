import torch.nn as nn
import types
def freeze(m):
    for p in m.parameters():
        p.requires_grad = False

def is_gate_param(name):
    return name.startswith("gate.")

def FrozenBatchNorm(base_module):
    """ Patches a module instance so that all `BatchNorm` child modules
        (1) have `requires_grad = False` in their weights
        (2) are always in "eval" mode

    Changes `base_module` in-place and returns `base_module`.
    """

    def train(self, mode=True):
        super(type(base_module), self).train(mode)
        if mode:
            for m in self.modules():
                if is_batch_norm(m):
                    m.train(False)

    base_module.train = types.MethodType(train, base_module)
    for m in base_module.modules():
        if is_batch_norm(m) and m.affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    # Freezing doesn't take effect until train() is called
    base_module.train(base_module.training)
    return base_module


def is_batch_norm(m):
    types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    if isinstance(m, type):
        return issubclass(m, types)
    else:
        return isinstance(m, types)


def is_weight_layer(cls):
    # Note: incomplete list
    types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    return issubclass(cls, types)


class FullyConnected(nn.Linear):
    """ A fully-connected layer. Flattens its input automatically.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        n = x.size(0)
        flat = x.view(n, -1)
        return super().forward(flat)
