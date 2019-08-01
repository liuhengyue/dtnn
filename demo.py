import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
import nnsearch.logging as mylog
import logging
from network.gated_cpm_mobilenet import GatedMobilenet, GatedStage
# dataset
import configparser
from dataloaders.cmu_hand_data import CMUHand
import random
from modules.utils import *



if __name__ == "__main__":
    # Logger setup
    mylog.add_log_level("VERBOSE", logging.INFO - 5)
    mylog.add_log_level("MICRO", logging.DEBUG - 5)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Need to set encoding or Windows will choke on ellipsis character in
    # PyTorch tensor formatting
    handler = logging.FileHandler("logs/demo.log", "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    root_logger.addHandler(handler)

    # order: "kernel_size", "stride", "padding", "nlayers", "nchannels", "ncomponents"
    backbone_stages = [GatedStage("conv", 3, 2, 0, 1, 32, 4), GatedStage("dw_conv", 3, 1, 1, 1, 64, 4),
                       GatedStage("dw_conv", 3, 2, 0, 1, 128, 4), GatedStage("dw_conv", 3, 1, 1, 1, 128, 4),
                       GatedStage("dw_conv", 3, 2, 0, 1, 256, 4), GatedStage("dw_conv", 3, 1, 1, 1, 256, 4),
                       GatedStage("dw_conv", 3, 1, 1, 1, 512, 4), GatedStage("dw_conv", 3, 1, 1, 1, 512, 4),
                       GatedStage("dw_conv", 3, 1, 1, 4, 512, 4), GatedStage("conv", 3, 1, 1, 1, 256, 4),
                       GatedStage("conv", 3, 1, 1, 1, 128, 4)]

    initial_stage = [GatedStage("conv", 3, 1, 1, 3, 128, 4), GatedStage("conv", 1, 1, 0, 1, 512, 4),
                     GatedStage("conv", 1, 1, 0, 1, 21, 1)]

    # fc_stage = GatedVggStage(1, 512, 2)

    gate = make_sequentialGate(backbone_stages, initial_stage)


    net = GatedMobilenet(gate, (3, 368, 368), 21, backbone_stages, None, initial_stage, [])
    gate_network = net.gate

    filename = model_file("ckpt/", 50)
    print(net.state_dict()["fn.87.bias"])
    with open( filename, "rb" ) as f:
      state_dict = torch.load(f, map_location="cpu")
      load_model( net, state_dict,
                  load_gate=True, strict=True)
      print(net.state_dict()["fn.87.bias"])
      # print(state_dict.keys())
    net.eval()
    # test_path = 'dataset/CMUHand/hand_labels/test/crop/Berry_roof_story.flv_000053_l.jpg'
    test_folder = 'dataset/CMUHand/hand_labels/test/crop'
    image_paths = glob.glob(os.path.join(test_folder, "*"))
    test_path = random.choice(image_paths)
    # print(image_paths)
    pred, frame = image_test(net, test_path, gated=True)
    # print(pred)
    kpts = get_kpts(pred)
    draw_paint(frame, kpts)

