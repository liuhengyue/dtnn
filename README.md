# throttling-demo

This is an initial README file for Latent AI throttle network demo.

The demo is an end-to-end trainable gated neural network focus on single-hand keypoints detection and dynamic hand gesture recognition. The whole network consists of two sub-networks: one is a variant [Global Context for Convolutional Pose Machines](https://arxiv.org/pdf/1906.04104.pdf) based on original paper  [Convolutional Pose Machines](https://arxiv.org/pdf/1602.00134.pdf). Its implementation can be found [here](https://github.com/Daniil-Osokin/gccpm-look-into-person-cvpr19.pytorch).

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


