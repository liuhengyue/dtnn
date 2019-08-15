import torch.nn as nn
from network.cpm_mobilenet import CPM_MobileNet
from network.gated_cpm_mobilenet import GatedMobilenet
from network.gated_c3d import GatedC3D, GatedStage
from modules.utils import *





class GestureNet():
    """
    The GestureNet network.
    For this network, all the init parameters should be seperate into two groups for the two sub-networks.
    """

    def __init__(self, num_refinement=3, dropout=0.0, gate_heatmap=False, weights_file=None, **kwargs):

        self.gate_heatmap = gate_heatmap
        self.heatmap_net_pars = self.make_heatmap_net()
        self.c3d_pars = self.make_c3d_net()
        if gate_heatmap:
            self.heatmap_net = GatedMobilenet(self.heatmap_net_pars["gate"], self.heatmap_net_pars["in_shape"], self.heatmap_net_pars["num_classes"],
                                          self.heatmap_net_pars["backbone"], None, self.heatmap_net_pars["initial"], [], dropout=dropout)
        else:
            self.heatmap_net = CPM_MobileNet(num_refinement)
        if weights_file is not None:
            self.heatmap_net.load_pretrained_weights(weights_file)
        self.heatmap_net.eval()
        self.set_no_grad()

        self.c3d_net = GatedC3D(self.c3d_pars["gate"], self.c3d_pars["in_shape"],
                                          self.c3d_pars["num_classes"],
                                          self.c3d_pars["c3d"], self.c3d_pars["fc"],
                                          dropout=dropout)

    def set_no_grad(self):
        for param in self.heatmap_net.parameters():
            param.requires_grad = False



    def get_heatmaps(self, x, u1=None):
        """
        C3D takes input shape: (BatchSize, 3, num_frames, 368, 368)
        Mobile_CPM takes input shape: (BatchSize, 3, 368, 368)
        x shape: (BatchSize, 3, num_frames, 368, 368)
        Should convert (BatchSize, 3, num_frames, 368, 368) ->
                       (BatchSize * num_frames, 3, 368, 368)
        """
        B, C, N, H, W = x.size()
        batch_heatmaps = []
        for b in range(B):
            x_slice = x[b] # (3, N, H, W)
            x_slice = x_slice.permute(1, 0, 2, 3) # (N, 3, H, W)
            if self.gate_heatmap:
                heatmap, _ = self.heatmap_net(x_slice, u1) # (N, 3 stages, 21, 45, 45)
            else:
                heatmap = self.heatmap_net(x_slice)
            # right now, just one stage so do not slice
            # heatmap_final_stage = heatmap
            heatmap_final_stage = heatmap[:, -1, :, :, :] # (N, 21, 45, 45)
            heatmap_final_stage = heatmap_final_stage.permute(1, 0, 2, 3) # (21, N, 45, 45)
            batch_heatmaps.append(heatmap_final_stage)

        batch_heatmaps = torch.stack(batch_heatmaps)

        return batch_heatmaps

    def forward(self, inputs, u=None):
        heatmaps = self.get_heatmaps(inputs, u)
        output, _ = self.c3d_net.forward(heatmaps, u)
        return output, heatmaps

    def eval(self):
        self.heatmap_net.eval()
        self.c3d_net.eval()

    def make_heatmap_net(self):
        # it should return a dict as the input to GestureNet
        #### first network: frome raw images to keypoint heatmaps
        backbone_stages = [GatedStage("conv", 3, 2, 0, 1, 32, 4), GatedStage("dw_conv", 3, 1, 1, 1, 64, 4),
                           GatedStage("dw_conv", 3, 2, 0, 1, 128, 4), GatedStage("dw_conv", 3, 1, 1, 1, 128, 4),
                           GatedStage("dw_conv", 3, 2, 0, 1, 256, 4), GatedStage("dw_conv", 3, 1, 1, 1, 256, 4),
                           GatedStage("dw_conv", 3, 1, 1, 1, 512, 4), GatedStage("dw_conv", 3, 1, 1, 1, 512, 4),
                           GatedStage("dw_conv", 3, 1, 1, 4, 512, 4), GatedStage("conv", 3, 1, 1, 1, 256, 4),
                           GatedStage("conv", 3, 1, 1, 1, 128, 4)]

        initial_stage = [GatedStage("conv", 3, 1, 1, 3, 128, 4), GatedStage("conv", 1, 1, 0, 1, 512, 4),
                         GatedStage("conv", 1, 1, 0, 1, 21, 1)]

        heatmap_stages = {"backbone": backbone_stages, "initial": initial_stage}
        gate = make_sequentialGate(heatmap_stages)
        in_shape = (3, 368, 368)
        num_classes = 21
        return {"backbone": backbone_stages, "initial": initial_stage, "gate": gate,
                "in_shape": in_shape, "num_classes": num_classes}

    def make_c3d_net(self):
        # it should return a dict as the input to GestureNet
        #### second network: from keypoint heatmaps to gestures
        c3d_stages = [GatedStage("conv", 3, 1, 1, 1, 64, 4), GatedStage("pool", (1, 2, 2), (1, 2, 2), 0, 1, 0, 0),
                      GatedStage("conv", 3, 1, 1, 1, 128, 2), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                      GatedStage("conv", 3, 1, 1, 2, 256, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                      GatedStage("conv", 3, 1, 1, 2, 512, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0),
                      GatedStage("conv", 3, 1, 1, 2, 512, 4), GatedStage("pool", 2, 2, 0, 1, 0, 0), ]

        fc_stages = [GatedStage("fc", 0, 0, 0, 2, 512, 2)]

        stages = {"c3d": c3d_stages, "fc": fc_stages}
        gate = make_sequentialGate(stages)
        in_shape = (21, 16, 45, 45)
        # in_shape = (3, 16, 368, 368) # for raw input
        num_classes = 5
        return {"c3d": c3d_stages, "fc": fc_stages, "gate": gate,
                "in_shape": in_shape, "num_classes": num_classes}