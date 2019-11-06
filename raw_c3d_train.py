import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchsummary import summary
import nnsearch.logging as mylog
import logging
import nnsearch.pytorch.gated.strategy as strategy
from nnsearch.pytorch.gated.module import GatedChainNetwork
from network.cpm_mobilenet import CPM_MobileNet
from network.gated_cpm_mobilenet import GatedMobilenet
from network.gated_c3d import GatedC3D, GatedStage
from modules.utils import *
# dataset
import configparser
# from dataloaders.cmu_hand_data import CMUHand
from tqdm import tqdm
from network.demo_model import GestureNet
from datetime import datetime
import re
from network.gated_c3d import C3dDataNetwork


def evaluate(elapsed_epochs, learner, testloader, cuda_devices=None):
    seed = 1
    # Hyperparameters interpret their 'epoch' argument as index of the current
    # epoch; we want the same hyperparameters as in the most recent training
    # epoch, but can't just subtract 1 because < 0 violates invariants.
    nclasses = len(testloader.dataset.class_names)
    class_correct = [0.0] * nclasses
    class_total = [0.0] * nclasses

    with torch.no_grad():
        learner.start_eval(elapsed_epochs, seed)
        for (batch_idx, data) in enumerate(tqdm(testloader)):
            images, labels = data
            if cuda_devices:
                images = images.cuda(cuda_devices[0])
                labels = labels.cuda(cuda_devices[0])
            log.debug("eval.images.shape: %s", images.shape)
            yhat = learner.forward(batch_idx, images, labels)
            log.debug("eval.yhat: %s", yhat)
            learner.measure(batch_idx, images, labels, yhat.data)
            _, predicted = torch.max(yhat.data, 1)
            log.debug("eval.labels: %s", labels)
            log.debug("eval.predicted: %s", predicted)
            c = (predicted == labels).cpu().numpy()
            log.debug("eval.correct: %s", c)
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

        learner.finish_eval(elapsed_epochs)
    for i in range(nclasses):
        if class_total[i] > 0:
            log.info("test %s '%s' : %s", elapsed_epochs, testloader.dataset.class_names[i],
                     class_correct[i] / class_total[i])
        else:
            log.info("test %s '%s' : None", elapsed_epochs, testloader.dataset.class_names[i])


if __name__ == "__main__":
    # Logger setup
    mylog.add_log_level("VERBOSE", logging.INFO - 5)
    mylog.add_log_level("MICRO", logging.DEBUG - 5)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Need to set encoding or Windows will choke on ellipsis character in
    # PyTorch tensor formatting
    experiment_name = 'demo_raw_c3d'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join("logs", experiment_name + '_' + timestamp + '.log')
    handler = logging.FileHandler(log_path, "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    root_logger.addHandler(handler)

    net = C3dDataNetwork((3, 16, 100, 160), num_classes=27)
    # init will all weights to zero
    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     # print(classname)
    #     if classname in ['Conv3d', 'FullyConnected']:
    #         nn.init.zeros_(m.weight.data)
    #         nn.init.zeros_(m.bias.data)
    # net.apply(weights_init)
    gate_network = net.gate
    ################### pre-trained
    pretrained = True
    if pretrained:
        # start = 12
        # filename = model_file("ckpt/gated_raw_c3d/", start, ".latest")
        filename = latest_checkpoints("ckpt/gated_raw_c3d/")[0]
        start = int(re.findall("\d+", os.path.basename(filename))[0])
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            load_model(net, state_dict, load_gate=True, strict=False)
    else:
        start = 0

    ### GPU support ###
    cuda = torch.cuda.is_available()
    # cuda = False
    device_ids = [1, 2, 3]
    # device_ids = [1]
    if cuda:
        net = net.cuda(device_ids[0])
        gate_network = gate_network.cuda(device_ids[0])
        if len(device_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=device_ids)
            gate_network = torch.nn.DataParallel(gate_network, device_ids=device_ids)
            print("Using multi-gpu: ", device_ids)
        else:
            print("Using single gpu: ", device_ids[0])
    ######################### dataset #######################
    # config = configparser.ConfigParser()
    # config.read('conf.text')
    # train_data_dir = config.get('data', 'train_data_dir')
    # train_label_dir = config.get('data', 'train_label_dir')
    batch_size = 20 * len(device_ids)
    # train_data = CMUHand(data_dir=train_data_dir, label_dir=train_label_dir)
    # train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # gesture dataset
    from dataloaders.dataset import VideoDataset
    subset = ['No gesture', 'Swiping Down', 'Swiping Up', 'Swiping Left', 'Swiping Right']
    subset = None
    train_data = VideoDataset(dataset='20bn-jester', split='train', clip_len=16, subset=subset)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data = VideoDataset(dataset='20bn-jester', split='val', clip_len=16, subset=subset)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    ######################### learner #######################
    # GatePolicyLearner
    import math
    import torch.optim as optim
    import nnsearch.pytorch.gated.learner as glearner
    lambda_gate = 1.0
    # learning_rate = 4e-5
    learning_rate = 5e-6
    # nclasses = 27
    # complexity_weights = []
    # for (m, in_shape) in net.gated_modules:
    #     complexity_weights.append(1.0) # uniform
    # lambda_gate = lambda_gate * math.log(nclasses)
    # optimizer = optim.SGD( net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4 )
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=1e-2, eps=1e-9, verbose=True)


    # gate_control = uniform_gate(0.9)
    gate_control = uniform_gate(0.0)
    # gate_control = uniform_gate(0.5)
    # gate_control = constant_gate(1.0)

    gate_loss = glearner.usage_gate_loss( penalty_fn)
    criterion = None
    learner = glearner.GatedDataPathLearner(net, optimizer, learning_rate,
                                            gate_network, gate_control, criterion=criterion, scheduler=scheduler)

    ######################### train #######################
    # start = 0
    train_epochs = 20
    n_utilization_stages = 10
    seed = 1
    eval_after_epoch = False
    u_stage_l = 0.0
    u_stage_r = 0.1
    increment = 0.1
    # u_stage_l = 0.0 if start == 0 else (start % train_epochs - 1 + 6.) / (train_epochs + 6.)
    for epoch in range(start, start + train_epochs):
        # u_stage starts from 0.1 up to 1.0
        # u_stage_r = (epoch + 11.) / (train_epochs + 11.)
        # u_stage_r = ((epoch - start) + 1.) / (train_epochs + 0)
        print("==== Train: Epoch %s: u_stage=[%s, %s]", epoch, u_stage_l, u_stage_r)
        log.info("==== Train: Epoch %s: u_stage=[%s, %s]", epoch, u_stage_l, u_stage_r)
        batch_idx = 0
        nbatches = math.ceil(len(train_data) / batch_size)
        learner.start_train(epoch, seed)
        running_corrects = 0.0
        running_loss = 0.0

        learner.update_gate_control(constant_gate(u_stage_r), u_stage=None) # (u_stage_l, u_stage_r))
        if (epoch - start + 1) % 2 == 0:
            u_stage_l = u_stage_r
            u_stage_r += increment
        for i, data in enumerate(tqdm(train_dataset)):
            inputs, labels = data
            if cuda:
                inputs = inputs.cuda(device_ids[0])
                labels = labels.cuda(device_ids[0])

            yhat = learner.forward(i, inputs, labels)
            loss = learner.backward(i, yhat, labels)
            probs = nn.Softmax(dim=1)(yhat)
            preds = torch.max(probs, 1)[1]
            batch_corrects = torch.sum(preds == labels.data).float()
            running_corrects += batch_corrects
            running_loss += loss.float()
            if i % 50 == 0:
                running_num = (i + 1) * batch_size
                step_loss = running_loss / running_num
                step_accuracy = running_corrects / running_num
                step_info = "Step [{}] loss: {:.4f}, accuracy: {:.4f}".format(i, step_loss, step_accuracy)
                print(step_info)
                log.info(step_info)

            batch_idx += 1
            # break
            # if i == 11:
            #     break
        end_info = "Epoch end, training accuracy: {:.4f}".format(running_corrects / len(train_data))
        print(end_info)
        log.info(end_info)
        learner.finish_train(epoch)
        learner.scheduler_step(loss, epoch)
        checkpoint(net, "ckpt/gated_raw_c3d", epoch + 1, learner)

        # eval
        if eval_after_epoch:
            evaluate(epoch, learner, test_dataset, cuda_devices=device_ids)

        # checkpoint(epoch + 1, learner)
        # Save final model if we haven't done so already
    # if args.train_epochs % args.checkpoint_interval != 0:
    #     checkpoint(start + args.train_epochs, learner, force_eval=True)
