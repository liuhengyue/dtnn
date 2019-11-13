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
from network.gated_c3d import C3dDataNetwork
from bandit_net import ContextualBanditNet
import json

def evaluate(u, results, learner, testloader, cuda_devices=None):
    seed = 1
    # Hyperparameters interpret their 'epoch' argument as index of the current
    # epoch; we want the same hyperparameters as in the most recent training
    # epoch, but can't just subtract 1 because < 0 violates invariants.
    nclasses = len(testloader.dataset.class_names)
    batch_size = testloader.batch_size
    class_correct = [0.0] * nclasses
    class_total = [0.0] * nclasses

    with torch.no_grad():
        learner.start_eval(u, seed)
        for (batch_idx, data) in enumerate(tqdm(testloader)):
            images, labels, indexes = data
            if cuda_devices:
                images = images.cuda(cuda_devices[0])
                labels = labels.cuda(cuda_devices[0])
            log.debug("eval.images.shape: %s", images.shape)
            yhat = learner.forward(batch_idx, images, labels)
            log.debug("eval.yhat: %s", yhat)
            # learner.measure(batch_idx, images, labels, yhat.data)
            probs = nn.Softmax(dim=1)(yhat)
            predicted = torch.max(probs, 1)[1]
            # _, predicted = torch.max(yhat.data, 1)
            log.debug("eval.labels: %s", labels)
            log.debug("eval.predicted: %s", predicted)
            c = (predicted == labels).cpu().numpy()
            # add to results
            predicted_list = predicted.tolist()
            labels_list = labels.cpu().tolist()
            for i, idx in enumerate(indexes.tolist()):
                if u == 0.1:
                    results[idx]['label'] = labels_list[i]
                results[idx]['prediction'][u] = predicted_list[i]
            log.debug("eval.correct: %s", c)
            # print("eval correct {}/{}".format(np.sum(c), batch_size))
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1


        learner.finish_eval(u)
    log.info("test u=%s, total %s [%s/%s]", u, sum(class_correct) / sum(class_total), sum(class_correct), sum(class_total))
    for i in range(nclasses):
        if class_total[i] > 0:
            log.info("'%s' : %s [%s/%s]", testloader.dataset.class_names[i],
                     class_correct[i] / class_total[i], class_correct[i], class_total[i])
        else:
            log.info("'%s' : None", testloader.dataset.class_names[i])


def controller_evaluate(data_network, controller_network, testloader, cuda_devices=None):
    indexes_list = []
    predicted_list = []
    labels_list = []
    u_list = []
    nclasses = len(testloader.dataset.class_names)
    batch_size = testloader.batch_size
    class_correct = [0.0] * nclasses
    class_total = [0.0] * nclasses
    ngate_levels = 10
    inc = 1.0 / ngate_levels
    u_s = torch.tensor([i * inc for i in range(1, ngate_levels + 1)], requires_grad=False).to(cuda_devices[0])
    with torch.no_grad():
        for (batch_idx, data) in enumerate(tqdm(testloader)):
            images, labels, indexes = data
            if cuda_devices:
                images = images.cuda(cuda_devices[0])
                labels = labels.cuda(cuda_devices[0])
            # forward controller
            output = controller_network(images)
            probs = torch.nn.Softmax(dim=1)(output)
            action = torch.argmax(probs, 1)
            u = torch.take(u_s, action)
            yhat, _ = data_network(images, u)
            # learner.measure(batch_idx, images, labels, yhat.data)
            probs = nn.Softmax(dim=1)(yhat)
            predicted = torch.max(probs, 1)[1]
            c = (predicted == labels).cpu().numpy()
            # add to results
            indexes_list += indexes.tolist()
            predicted_list += predicted.tolist()
            labels_list += labels.cpu().tolist()
            u_list += u.tolist()

            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1

    log.info("test u=%s, total %s [%s/%s]", u, sum(class_correct) / sum(class_total), sum(class_correct),
             sum(class_total))
    for i in range(nclasses):
        if class_total[i] > 0:
            log.info("'%s' : %s [%s/%s]", testloader.dataset.class_names[i],
                     class_correct[i] / class_total[i], class_correct[i], class_total[i])
        else:
            log.info("'%s' : None", testloader.dataset.class_names[i])

    results = {"id": indexes_list, "label": labels_list, "prediction": predicted_list, "u": u_list}
    return results
if __name__ == "__main__":
    # Logger setup
    mylog.add_log_level("VERBOSE", logging.INFO - 5)
    mylog.add_log_level("MICRO", logging.DEBUG - 5)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Need to set encoding or Windows will choke on ellipsis character in
    # PyTorch tensor formatting
    experiment_name = 'eval_raw_c3d'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join("logs", experiment_name + '_' + timestamp + '.log')
    results_path = os.path.join("logs", experiment_name + '_' + timestamp + '.json')
    handler = logging.FileHandler(log_path, "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    root_logger.addHandler(handler)

    net = C3dDataNetwork((3, 16, 100, 160), num_classes=27)
    gate_network = net.gate
    ################### must load the model to eval
    # start = 11
    # filename = model_file("ckpt/gated_raw_c3d/", start, ".latest")
    filename = latest_checkpoints("ckpt/gated_raw_c3d/")[0]

    with open(filename, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        load_model(net, state_dict, load_gate=True, strict=True)
        load_info = "Load weights from {}".format(filename)
        print(load_info)
        log.info(load_info)

    # add controller
    control = True
    if control:
        controller = ContextualBanditNet()
        controller.eval()
        filename = latest_checkpoints("ckpt/controller/", prefix="controller")[0]
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            load_model(controller, state_dict, load_gate=True, strict=True)
            load_info = "Load weights from {}".format(filename)
            print(load_info)
            log.info(load_info)

    ### GPU support ###
    cuda = torch.cuda.is_available()
    # cuda = False
    device_ids = [0, 1, 2, 3] if cuda else None
    # device_ids = [1]
    if cuda:
        net = net.cuda(device_ids[0])
        gate_network = gate_network.cuda(device_ids[0])
        if control:
            controller = controller.cuda(device_ids[0])
        if len(device_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=device_ids)
            gate_network = torch.nn.DataParallel(gate_network, device_ids=device_ids)
            if control:
                controller = torch.nn.DataParallel(controller, device_ids=device_ids)
            print("Using multi-gpu: ", device_ids)
        else:
            print("Using single gpu: ", device_ids[0])
    ######################### dataset #######################
    # config = configparser.ConfigParser()
    # config.read('conf.text')
    # train_data_dir = config.get('data', 'train_data_dir')
    # train_label_dir = config.get('data', 'train_label_dir')
    batch_size = 50 * len(device_ids) if cuda else 1

    # train_data = CMUHand(data_dir=train_data_dir, label_dir=train_label_dir)
    # train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # gesture dataset
    from dataloaders.dataset import VideoDataset
    subset = ['No gesture', 'Swiping Down', 'Swiping Up', 'Swiping Left', 'Swiping Right']
    subset = None
    test_data = VideoDataset(dataset='20bn-jester', split='val', clip_len=16, subset=subset)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    ######################### learner #######################
    # GatePolicyLearner
    import math
    import torch.optim as optim
    import nnsearch.pytorch.gated.learner as glearner
    lambda_gate = 1.0
    learning_rate = 4e-5
    # nclasses = 27
    # complexity_weights = []
    # for (m, in_shape) in net.gated_modules:
    #     complexity_weights.append(1.0) # uniform
    # lambda_gate = lambda_gate * math.log(nclasses)
    # optimizer = optim.SGD( net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4 )
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=1e-2, eps=1e-9, verbose=True)


    # gate_control = uniform_gate()
    gate_control = constant_gate(0.0)

    gate_loss = glearner.usage_gate_loss( penalty_fn)
    criterion = None
    learner = glearner.GatedDataPathLearner(net, optimizer, learning_rate,
                                            gate_network, gate_control, criterion=criterion, scheduler=scheduler)

    ######################### eval #######################
    # u_grid = [0.8, 0.85, 0.9, 0.95]
    # u_grid = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    u_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # u_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    # larger data structure for storing all the predictions and labels
    # {
    #   0 : {'label' : 1 , 'prediction' : { 0.1 : 0, 0.2 : 1, ...} },
    #   1 : {...},
    #   ...
    # }
    from collections import defaultdict
    results = defaultdict(dict)
    n_examples = len(test_data)

    # print(results)
    if control:
        results = controller_evaluate(net, controller, test_dataset, cuda_devices=device_ids)
    else:
        for i in range(n_examples):
            results[i]['label'] = 0
            results[i]['prediction'] = {u: 0 for u in u_grid}
        for i, u in enumerate(u_grid):
            print("==== Eval for u = %s ====", u)
            log.info("==== Eval for u = %s ====", u)
            learner.update_gate_control(constant_gate(u))
            evaluate(u, results, learner, test_dataset, cuda_devices=device_ids)

    with open(results_path, 'w') as f:
        json.dump(results, f)
        # checkpoint(epoch + 1, learner)
        # Save final model if we haven't done so already
    # if args.train_epochs % args.checkpoint_interval != 0:
    #     checkpoint(start + args.train_epochs, learner, force_eval=True)
