import os

import ipdb
import matplotlib
from tqdm import tqdm
import logging
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from model.faster_rcnn_densenet import FasterRCNNDensenet
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import argparse
import sys
#sys.path.append('/home/local/SRI/e29288/AC/nnsearch')

#import nnsearch
import nnsearch.logging as mylog

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')





def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(opt): #train(**kwargs):
    #opt._parse(kwargs)

    dataset = Dataset(opt)
    logging.info('Loading data...')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    logging.debug('Data loaded')

    if opt.model == 'gated-vgg':
        faster_rcnn = FasterRCNNVGG16(opt=opt, gated=True)
    elif opt.model == 'densenet':
        faster_rcnn = FasterRCNNDensenet(opt=opt, gated=False) 
    elif opt.model == 'gated-densenet':
        faster_rcnn = FasterRCNNDensenet(opt=opt, gated=True)
    else: # opt.model == 'vgg'
        faster_rcnn = FasterRCNNVGG16(opt=opt, gated=False)

    logging.debug('Model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.model_pretrain_path:
        trainer.load(opt.model_pretrain_path)
        logging.info('Load pretrained model from %s' % opt.model_pretrain_path)

    #trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.learning_rate
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:

                # rpn confusion matrix(meter)
                
                logging.info(str(trainer.get_meter_data()))
                #logging.info(str(trainer.rpn_cm.value().tolist()))
                #logging.info(str(at.totensor(trainer.roi_cm.conf, False).float()))

                #trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(aws_sync= opt.aws_sync, save_path = opt.output, best_map=best_map, epoch=epoch)
        if epoch % 9== 0:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

#        trainer.vis.plot('test_map', eval_result['map'])
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        logging.info(log_info)
        # trainer.vis.log(log_info)
        if epoch == 13: 
            break


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train FasterRCNN')
    parser.add_argument('--voc-data-dir', '-vdd', type=str, default='/data1/nnsearch/data/VOCdevkit/VOC2007/')
    parser.add_argument('--min-size', type=int, default=600)
    parser.add_argument('--max-size', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--test-num-workers', type=int, default=8)
    parser.add_argument('--rpn-sigma',default=3.)
    parser.add_argument('--roi-sigma', default=1.)
    parser.add_argument('--weight-decay', type=float, default=0.005)
    parser.add_argument('--lr-decay', type=float, default=0.1)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--env', type=str, default='faster-rcnn')
    parser.add_argument('--port','-p', type=int, default=8097)
    parser.add_argument('--plot-every', type=int, default=40)
    parser.add_argument('--data', '-d', type=str, default='voc')
    parser.add_argument('--model', type=str, default='densenet', choices=['vgg', 'gated-vgg', 'densenet', 'gated-densenet'])
    parser.add_argument('--epoch', type=int, default=14)
    parser.add_argument('--use-adam', action='store_true')
    parser.add_argument('--use-drop', action='store_true')
    parser.add_argument('--use-chainer', action='store_true')
    parser.add_argument('--debug-file', type=str, default='/tmp/debugf')
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--torch-pretrain', action='store_true')
    parser.add_argument('--model-pretrain-path', type=str, default=None)
    parser.add_argument('--output', type=str, default='/data1/nnsearch/checkpoints/faster-rcnn')
    parser.add_argument('--log-level',default='VERBOSE', choices=['VERBOSE', 'DEBUG', 'INFO'])
    parser.add_argument( "--aws-sync", type=str, default=None,
                help="If not None, a shell command to run after every epoch to sync results." )

    opt = parser.parse_args()

    
    #mylog.add_log_level( "MICRO",   logging.DEBUG - 5 )
    #mylog.add_log_level( "NANO",    logging.DEBUG - 6 )
    #mylog.add_log_level( "PICO",    logging.DEBUG - 7 )
    #mylog.add_log_level( "FEMTO",   logging.DEBUG - 8 )
    mylog.add_log_level( "VERBOSE", logging.INFO - 5 )
    root_logger = logging.getLogger()
    root_logger.setLevel( mylog.log_level_from_string( opt.log_level ) )
    handler = logging.FileHandler(os.path.join( opt.output, "faster_rcnn.log" ), "w", "utf-8")
    handler.setFormatter( logging.Formatter("%(levelname)s:%(name)s: %(message)s") )
    root_logger.addHandler(handler)
    # logging.basicConfig()
    # logging.getLogger().setLevel( logging.INFO )
    log = logging.getLogger( __name__ )
    log.info( "Git revision: %s", mylog.git_revision() )
    log.info(opt)




    train(opt)

