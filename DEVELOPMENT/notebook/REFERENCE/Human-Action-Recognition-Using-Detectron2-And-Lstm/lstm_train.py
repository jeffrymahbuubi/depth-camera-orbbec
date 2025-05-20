#!/usr/bin/env python
# coding: utf-8

# **Get the training dataset**

# In[ ]:


get_ipython().system('wget -O RNN-HAR-2D-Pose-database.zip https://drive.google.com/u/1/uc?id=1IuZlyNjg6DMQE3iaO1Px6h1yLKgatynt')


# In[ ]:


get_ipython().system('unzip RNN-HAR-2D-Pose-database.zip')


# **List first two rows of training set**

# In[ ]:


get_ipython().system('head -2 /content/RNN-HAR-2D-Pose-database/X_train.txt')


# **Clone the codebase**

# In[ ]:


get_ipython().system('git clone https://github.com/spmallick/learnopencv.git')


# In[ ]:


get_ipython().run_line_magic('cd', 'learnopencv/Human-Action-Recognition-Using-Detectron2-And-Lstm')


# **Install dependencies**

# In[ ]:


get_ipython().system('pip install -r requirements.txt')


# In[7]:


DATASET_PATH = "/content/RNN-HAR-2D-Pose-database/"


# In[8]:


from argparse import ArgumentParser

def configuration_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--data_root', type=str, default=DATASET_PATH)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_class', type=int, default=6)
    return parser


# In[9]:


import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from src.lstm import ActionClassificationLSTM, PoseDataModule


# In[10]:


def do_training_validation():
    pl.seed_everything(21)    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = configuration_parser(parser)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # init model    
    model = ActionClassificationLSTM(34, 50, learning_rate=args.learning_rate)
    data_module = PoseDataModule(data_root=args.data_root,
                                        batch_size=args.batch_size)    
    #save only the top 1 model based on val_loss
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='step')  
    #trainer
    trainer = pl.Trainer.from_argparse_args(args,
        # fast_dev_run=True,
        max_epochs=args.epochs, 
        deterministic=True, 
        gpus=1, 
        progress_bar_refresh_rate=1, 
        callbacks=[EarlyStopping(monitor='train_loss', patience=15), checkpoint_callback, lr_monitor])    
    trainer.fit(model, data_module)    
    return model


# In[ ]:


# To reload tensorBoard
get_ipython().run_line_magic('load_ext', 'tensorboard')

# logs folder path
get_ipython().run_line_magic('tensorboard', '--logdir=lightning_logs')


# **Training**
# 
# 

# In[ ]:


do_training_validation()


# **Get the saved model**

# In[13]:


import os
def get_latest_run_version_ckpt_epoch_no(lightning_logs_dir='lightning_logs', run_version=None):
    if run_version is None:
        run_version = 0
        for dir_name in os.listdir(lightning_logs_dir):
            if 'version' in dir_name:
                if int(dir_name.split('_')[1]) > run_version:
                    run_version = int(dir_name.split('_')[1])                
    checkpoints_dir = os.path.join(lightning_logs_dir, 'version_{}'.format(run_version), 'checkpoints')    
    files = os.listdir(checkpoints_dir)
    ckpt_filename = None
    for file in files:
        print(file)
        if file.endswith('.ckpt'):
            ckpt_filename = file        
    if ckpt_filename is not None:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
    else:
        print('CKPT file is not present')    
    return ckpt_path


# In[ ]:


ckpt_path = get_latest_run_version_ckpt_epoch_no()
print('The latest model path: {}'.format(ckpt_path))

