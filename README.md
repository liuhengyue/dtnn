# throttling-demo

Latent AI throttle network demo.

The demo is an end-to-end trainable throttleable neural network for dynamic hand gesture recognition. It consists of three main conponents: gated 3D Convolutional Neural Networks (C3D) data network for gesture recognition of five classes [Swiping left/right/up/down, no gesture], a small controller network providing utilization parameter for the data network, and single-hand keypoint estimation network as a add-on module. The demo also has a simple OpenCV visualization.


## Usage
Download the pretrained weights from [latentai > sample-models > demo branch](https://gitlab.com/latentai/sample-models/tree/demo/demo). 

Put `controller_network_2.pkl.latest` inside folder  `ckpt/controller`. 

Put `cpm_r3_model_epoch1540.pth` inside `ckpt/`. 

Put `model_110.pkl.latest` inside `ckpt/gated_raw_c3d`.

Run `pip install -r requirements.txt` to install dependencies.

To run the demo, just clone this repo, then go in the folder through terminal or any IDE of your choice, run `python demo.py`. By default, it will use one GPU and enable all the features. I have not write a requirements file for dependencies yet. 

## Throttleable Neural Network

The concepts and implementations take from this paper [Toward Runtime-Throttleable Neural Networks](https://arxiv.org/abs/1905.13179). For the specific implementation of demo, several attempts have been made. TNN requries convolutional and dense layers to be gated, so gated networks were implemented first. In this repo, there are a gated keypoint detection framework and gated C3D implemented. The experiments are mainly on widthwise nested sequential gates while other options are also available.

## Gesture Recognition

Gesture recognition framework is based on a basic [C3D](https://arxiv.org/abs/1412.0767) network with gated version implemented. Some implementations are from [here](https://github.com/jfzhang95/pytorch-video-recognition). Only five gesture classes are used.

The detailed structure is listed here: (todo)

## Hand Keypoint Estimation

Currently, the single-hand keypoints detection model is just used for hand keypoints visualization, but the contextual information can be used for building a data-driven controller. The implementation is based on a variant [Global Context for Convolutional Pose Machines](https://arxiv.org/pdf/1906.04104.pdf) of the original paper  [Convolutional Pose Machines](https://arxiv.org/pdf/1602.00134.pdf). Its implementation can be found [here](https://github.com/Daniil-Osokin/gccpm-look-into-person-cvpr19.pytorch). Part of the implementations can also be found [here](https://github.com/HowieMa/CPM_Pytorch).

## File Structure

The project file structure is listed here:

| Folder/File | Description | Used in Demo |
| ----------- | ----------- | ------------ |
| ckpt/ | Store the trained model files and checkpoints. | Yes |
| dataloaders/  |   Scripts to pre-process and load different datasets.| No |
| dataset/   |    Store different datasets.| No |
| logs/ |       Store trainning and evaluation logs.| No |
| modules/ |    Inside, `utils.py` contains utility functions for gated network. Others are for keypoint estimation.| Yes |
| network/ |  Different implementations of neural networks (gated and non-gated).| Yes |
| nnsearch/ | Jesse's codes for throttleable NN (some changes are made). | Yes |
| src/ | Some functions for displaying keypoint heatmap. | No |
| visualization/ | Store local visualization outputs, images, etc. | No |
| bandit_net.py | Controller network implementation. | Yes |
| c3d_train.py | Original training script for video action recognition.| No |
| conf.text | Configuration file for training/testing keypoint estimation. | No |
| cpm_*.py | Keypoint estimation related scripts; the trained model is used for demo visualization. | No |
| demo.py | The demo entry point. | Yes |
| demo_train.py | Deprecated. This old demo structure uses keypoint heatmaps for gesture recognition. | No |
| demo_train_cpm.py | Deprecated. This is for training gated CPM which is not working as expected. | No |
| gate_test.py | Script for checking gated layer implementation. | No |
| inference.py | Original inference script for video action recognition. | No |
| model_util.py | Samyak's code copy from Jesse's nnsearch/ folder. | No |
| mypath.py | Training configuration and paths for gesture recognition. | No |
| policy.py | Samyak's code copy from Jesse's nnsearch/ folder on RL policy. | No |
| qnet.py | Samyak's code example for contextual Q learning network. | No |
| raw_c3d_*.py | The training/evaluation scripts for the demo's gesture recognition part. | Train |
| shape_flop_util.py | Probably deprecated. Functions to Calculate the flops of defined layers.| No |
| testthrottle.py | Deprecated. Samyak's original codes for controller. | No |
| throttle_train.py | For training the controller network; currently not working well. | No |
| train.py | Probably same with c3d_train.py. Original training script for video action recognition.| No |
| weights_experiment.py | Speed test for gated structure using all zero/random weights on CPU/GPU. | No |




## Dataset

The dataset for training the keypoint estimation is from [CMU Hand Dataset](http://domedb.perception.cs.cmu.edu/handdb.html), both real and synthetic dataset are used.

The dataset for training the gesture recognition model is from [20BN-jester V1](https://20bn.com/datasets/jester).

## Demo

![](demo.gif)




