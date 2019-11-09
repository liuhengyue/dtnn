import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model
from network import cpm_c3d_model

# Use GPU if available else revert to CPU
cuda = torch.cuda.is_available()
devices = [0, 1, 2, 3]
print("Devices being used:", devices)

nEpochs = 50  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = False # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 10 # Store a model every snapshot epochs
lr = 1e-3 # Learning rate
pretrained = False

dataset = '20bn-jester' # Options: hmdb51 or ucf101 or 20bn-jester

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
elif dataset == '20bn-jester':
    num_classes = 27
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

def filter_none_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter (lambda x:x[0] is not None and x[1] is not None, batch))
    if batch == []:
        return []
    # if list has only one element, it should be also fine
    collate = torch.utils.data.dataloader.default_collate(batch)
    return collate

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=pretrained)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif modelName == 'CPM-C3D':
        model = cpm_c3d_model.CPM_C3D(num_classes=num_classes, pretrained_cpm=True, pretrained_c3d=pretrained)
        cpm_c3d_model.set_no_grad(model)
        train_params = cpm_c3d_model.get_trainable_params(model)
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # for i, p in enumerate(train_params):
        #     if p.requires_grad:
        #         print(i)
        # return

    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.cuda(devices[0])
    criterion.to(devices[0])
    # multi gpu support
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=devices)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataset = VideoDataset(dataset=dataset, split='train',clip_len=16, subset=None)
    val_dataset = VideoDataset(dataset=dataset, split='val', clip_len=16, subset=None)
    test_dataset = VideoDataset(dataset=dataset, split='test', clip_len=16, subset=None)


    # or just try a subset
    # train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=200)
    # val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=200)
    train_sampler, val_sampler = None, None
    train_shuffle = True
    batch_size = 22 * len(devices)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=train_shuffle, collate_fn=filter_none_collate)
    val_dataloader   = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, collate_fn=filter_none_collate)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                # scheduler.step()
                model.train()
            else:
                if epoch % 5 != 0:
                    break
                model.eval()

            for data in tqdm(trainval_loaders[phase]):
                # if it did not load the dataset successfully for some images, skip
                if data == []:
                    continue
                inputs, labels, _ = data
                if cuda:
                    # move inputs and labels to the device the training is taking place on
                    inputs = inputs.cuda(devices[0])
                    labels = labels.cuda(devices[0])
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            if isinstance(model, torch.nn.DataParallel):
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.cuda(devices[0])
                labels = labels.cuda(devices[0])

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()
