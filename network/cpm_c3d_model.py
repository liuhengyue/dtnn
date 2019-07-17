import torch
import torch.nn as nn
import torch.nn.functional as F
from mypath import Path
from torchsummary import summary
# from network.cpm import CPM
from network.cpm_mobilenet import CPM_MobileNet
from network.C3D_model import C3D

class CPM_C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained_c3d=False, pretrained_cpm=True):
        super(CPM_C3D, self).__init__()
        self.cpm = CPM_MobileNet(2)
        self.c3d = C3D(num_classes=num_classes, pretrained=pretrained_c3d)
        # TODO check if it only initialize 3d layers
        self.__init_weight()
        if pretrained_cpm:
            self.__load_pretrained_cpm_weights()
            # freeze the weights
            self.cpm.eval()
        if pretrained_c3d:
            self.__load_pretrained_weights()

    def __load_pretrained_cpm_weights(self):
        cuda = torch.cuda.is_available()
        # saved model dict
        state_dict = torch.load(Path.cpm_model_dir(), lambda storage, loc: storage)
        # with open('state_dict.txt', 'w') as f:
        #     for item in state_dict.keys():
        #         f.write("%s\n" % item)
        # defined model dict
        s_dict = self.state_dict()
        # with open('loaded_state_dict.txt', 'w') as f:
        #     for item in s_dict.keys():
        #         f.write("%s\n" % item)
        # print(len(state_dict), len(s_dict))
        # print(state_dict.keys())
        if cuda:
            # TODO: have not test cuda dict matching
            self.load_state_dict(state_dict)
        else:
            # trained with DataParallel but test on cpu
            # single_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module", "cpm") # remove `module.`
                if name in s_dict.keys():
                    s_dict[name] = v
            # load params
            self.load_state_dict(s_dict)


    def forward(self, x):
        """
        C3D takes input shape: (BatchSize, 3, num_frames, 368, 368)
        CPM takes input shape: (BatchSize, 3, 368, 368), (BatchSize, 368, 368)
        Should convert (BatchSize, 3, num_frames, 368, 368) ->
                       (BatchSize * num_frames, 3, 368, 368)
        """
        B, C, N, H, W = x.size()
        x_cpm = x.permute(0, 2, 1, 3, 4)
        # print(x_cpm.size())
        x_cpm = x_cpm.contiguous().view(-1, C, H, W)
        heatmap = self.cpm(x_cpm)
        heatmap_final_stage = heatmap[:,-1,:,:,:]
        # _, _, num_kpts, heatmap_h, heatmap_w = heatmap.size()
        # heatmap = heatmap.view(B, -1, heatmap_h, heatmap_w)
        # heatmap_final_stage shape: (B, 21, 45, 45)
        # print(heatmap_final_stage.size())
        # need this if two networks size mismatch
        # heatmap_final_stage_upscaled = F.interpolate(heatmap_final_stage, size=(112, 112)) 
        # heatmap_final_stage_upscaled.unsqueeze_(2)
        heatmap_final_stage.unsqueeze_(2)

        logits = self.c3d(heatmap_final_stage)

        return heatmap, logits

    # def forward(self, x, c):
    #     """
    #     C3D takes input shape: (BatchSize, 3, num_frames, 368, 368)
    #     CPM takes input shape: (BatchSize, 3, 368, 368), (BatchSize, 368, 368)
    #     Should convert (BatchSize, 3, num_frames, 368, 368) ->
    #                    (BatchSize * num_frames, 3, 368, 368)
    #     """
    #     B, C, N, H, W = x.size()
    #     x_cpm = x.permute(0, 2, 1, 3, 4)
    #     x_cpm = x_cpm.view(-1, C, H, W)
    #     print(x_cpm.size())
    #     print(c.size())
    #     heatmap = self.cpm(x_cpm, c)
    #     logits = self.c3d(x)

    #     return heatmap, logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def set_no_grad(model, submodel='cpm'):
    if submodel == 'cpm':
        for name, param in model.named_parameters():
            if submodel in name:
                param.requires_grad = False

def get_trainable_params(model):
    for param in model.parameters():
        if param.requires_grad:
            yield param


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 1, 368, 368)
    c = torch.randn(1, 368, 368)
    net = CPM_C3D(num_classes=27, pretrained_cpm=True, pretrained_c3d=False)
    # print(net)
    # summary(net, [(3, 1, 368, 368), (368, 368)] )
    summary(net, (3, 1, 368, 368))
    # heatmap, output = net.forward(inputs, c)
    # print(heatmap.size())
    # print(output.size())