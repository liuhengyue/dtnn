# throttling-demo

Latent AI throttle network demo.

The demo is an end-to-end trainable throttleable neural network for dynamic hand gesture recognition. It consists of three main conponents: gated 3D Convolutional Neural Networks (C3D) data network for gesture recognition of five classes [Swiping left/right/up/down, no gesture], a small controller network providing utilization parameter for the data network, and single-hand keypoint estimation network as a add-on module. The demo also has a simple OpenCV visualization.


## Usage

To run the demo, just clone this repo, then go in the folder through terminal or any IDE of your choice, run `python demo.py`. By default, it will use one GPU and enable all the features. I have not write a requirements file for dependencies yet. 

## Throttleable Neural Network

The concepts and implementations take from this paper [Toward Runtime-Throttleable Neural Networks](https://arxiv.org/abs/1905.13179). For the specific implementation of demo, several attempts have been made. TNN requries convolutional and dense layers to be gated, so gated networks were implemented first. In this repo, there are a gated keypoint detection framework and gated C3D implemented. The experiments are mainly on widthwise nested sequential gates while other options are also available.

## Gesture Recognition

Gesture recognition framework is based on a basic [C3D](https://arxiv.org/abs/1412.0767) network with gated version implemented. Some implementations are from [here](https://github.com/jfzhang95/pytorch-video-recognition). The detailed structure is listed here: (todo)

## Hand Keypoint Estimation

Currently, the single-hand keypoints detection model is just used for hand keypoints visualization, but the contextual information can be used for building a data-driven controller. The implementation is based on a variant [Global Context for Convolutional Pose Machines](https://arxiv.org/pdf/1906.04104.pdf) of the original paper  [Convolutional Pose Machines](https://arxiv.org/pdf/1602.00134.pdf). Its implementation can be found [here](https://github.com/Daniil-Osokin/gccpm-look-into-person-cvpr19.pytorch).

## File Structure

The project file structure is listed here:

ckpt/        Store the trained model files and checkpoints.
dataloaders/  Scripts to pre-process and load different datasets.
dataset/     Store different datasets.
logs/       Store trainning and evaluation logs.
modules/    Inside, `utils.py` is probably the most important file, which contains utility functions for gated network. Others are for keypoint estimation.
network/    Different implementations of neural networks (gated and non-gated)

## Convolutional Pose Machines 

This is the Pytorch

There are 7 files in this folder

--handpose_data_cpm.py    
data loader for Hand Pose dataset
    
--handpose_no_label.py  
data loader for Hand Pose dataset without ground truth   
    
--cpm.py   
Pytorch cpm model 

--train.py    
--test.py 
--save.py  
--predict.py  
--conf.text         


## usage 
#### 1 train model  
    python cpm_train.py   

You may revise the variable in  **conf.text**    

>train_data_dir  =   
train_label_dir =   
learning_rate   = 8e-6     
batch_size      = 16   
epochs          = 50   
begin_epoch     = 0   

Thus change the path to your own datasets and  train CPM on your own 
REMEMBER that you may implement new data loader for you own datasets. 

After this, you will get models for several epoches.
The models are saved in folder **ckpt/**  like 
> ckpt/model_epoch10.pth 
      

 
#### 2 test model   
    python cpm_Test.py         
    
After running this, you will get PCK score for each epoch  
You can select the best trained models
  
#### 3 save prediction results 
    python cpm_save.py    

After step 2, you will know which is the best epoch, 
thus you should revise conf.text and change the value of **best_model**
 
    
#### 4 apply models on datasets without ground truth
    python cpm_predict.py    

This step is for applying trained model on datasets without ground truth 


## 


