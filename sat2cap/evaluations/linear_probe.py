import os
from PIL import Image 
from imageio import imread
import code
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import random
from argparse import ArgumentParser, RawTextHelpFormatter
import ssl
#pytorch imports
import torch
import torch.nn as nn
import pytorch_lightning as pl 
from torchvision.datasets import ImageFolder
import torchvision
import torchgeo

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics import Accuracy
#huggingface import
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import rasterio

##local imports
from ..models.geomoco import GeoMoCo
from ..utils import random_seed, load_model
from ..data import landuse

class LinearProbe(pl.LightningModule):
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.vit_map = {'32':'openai/clip-vit-base-patch32', '16':'openai/clip-vit-base-patch16', '14L':'openai/clip-vit-large-patch14'}
        
        if self.hparams.model_status.lower()=='clip':
            print('Training for normal CLIP')
            self.model = load_model.load_clip(self.hparams.ckpt_path).imo_encoder
        
        elif self.hparams.model_status.lower()=='geoclip':
            print('Training for pretrained GeoCLIP')
            self.model = load_model.load_geomoco(self.hparams.ckpt_path).imo_encoder
        
        self.imo_encoder = self.model.eval()

        for param in self.imo_encoder.parameters():
            param.requires_grad=False
        
        self.fc1 = nn.Linear(512, self.hparams.num_classes)
        self.soft = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        
        
    def forward(self, batch):
        x,y = batch
        normalized_imo_embeddings,_ = self.imo_encoder.forward(x)
        #unnormalized_imo_embeddings = unnormalized_imo_embeddings
        output = self.fc1(normalized_imo_embeddings)
        return output

    def shared_step(self,batch):
        x,y = batch
        output = self(batch)
        loss = self.loss(output, y)
        acc = self.acc(output, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss,acc = self.shared_step(batch)
        self.log('loss', loss, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log('acc', acc, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss,acc = self.shared_step(batch)
        self.log('val_loss', loss, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log('val_acc', acc, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss,acc = self.shared_step(batch)
        self.log('test_loss', loss, sync_dist=True,on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('test_acc', acc, sync_dist=True,on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss, acc

    def configure_optimizers(self):
        print(f'Initializing Learning rate {self.hparams.learning_rate}')
        params = list(filter(lambda p:p.requires_grad, self.parameters())) #+ list(self.logit_scale)
        self.optim = torch.optim.AdamW(params,
            lr=self.hparams.learning_rate)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = self.optim, T_0=self.hparams.warmup_epochs)
        return {'optimizer': self.optim, 
        'lr_scheduler': self.scheduler
        }


def tif_to_pil(tif):
    return rasterio.open(tif)

def get_args():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    
    #other params
    parser.add_argument('--train_size', type=float, default=0.3)
    parser.add_argument('--dataset_name', type=str, default='resisc')
    parser.add_argument('--model_status', type=str, default='clip')
    parser.add_argument('--ckpt_path', type=str, default='root_path/logs/GeoClip/rmlo6lic/checkpoints/step=35500-val_loss=5.013.ckpt')
    #trainer hparams
    parser.add_argument('--batch_size', type=str, default=64)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--mode', type=str, default='dev')

    #wandb hparams
    parser.add_argument('--log_dir', type=str, default='root_path/logs')
    parser.add_argument('--project_name', type=str, default='SupervisedEval')
    parser.add_argument('--run_name', type=str, default='first_attempt')
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--wandb_resume', type=str, default='none')

    #model hparams
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    #test hparams
    parser.add_argument('--classification_ckpt', type=str, default='root_path/logs/SupervisedEval/xs6lkjwo/checkpoints/epoch=92-step=109833-val_acc=0.889.ckpt')
    

    args = parser.parse_args()
    return args

def main(args):
    #get dataloaders
    train_dataloader, val_dataloader, test_dataloader, num_classes = landuse.get_dataloader(args.dataset_name, args.train_size, args.batch_size, args.num_workers)
    args.num_classes = num_classes
    
    #set loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    if args.wandb_resume.lower()=='none':
        wb_logger = WandbLogger(save_dir=args.log_dir,project=args.project_name, name=args.run_name, mode=args.wandb_mode)
    else:
        wb_logger = WandbLogger(save_dir=args.log_dir,project=args.project_name, mode=args.wandb_mode, resume=args.wandb_resume)
    #, resume=args.wandb_resume 
    ckpt_monitors = ((
            ModelCheckpoint(monitor='val_loss', filename='{step}-{val_loss:.3f}', mode='min', save_top_k=2, save_last=True),
                ModelCheckpoint(monitor='val_acc',filename='{epoch}-{step}-{val_acc:.3f}', mode='max', save_top_k=2, save_last=True)
        ))

    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = pl.Trainer(fast_dev_run=15, max_epochs=4, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=1,
        accelerator=args.accelerator, devices=args.devices, callbacks=[*ckpt_monitors, lr_logger])
    
    elif args.mode == 'train':
        print('Training Run')
        trainer = pl.Trainer(precision='32', max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=1, 
        accelerator=args.accelerator, devices=args.devices, callbacks=[*ckpt_monitors, lr_logger], check_val_every_n_epoch=1, log_every_n_steps=15)
        model = LinearProbe(args)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    elif args.mode =='test':
        print('Test Run')
        trainer = pl.Trainer(precision='32', devices=1, accelerator='gpu')
        classification_model_ckpt = torch.load(args.classification_ckpt)
        classification_model = LinearProbe(hparams = classification_model_ckpt['hyper_parameters'])
        classification_model.load_state_dict(classification_model_ckpt['state_dict'])
        test_results = trainer.test(classification_model, dataloaders=test_dataloader)
       # code.interact(local=dict(globals(), **locals()))

    else:
        raise ValueError('Invalid value for mode')

        
if __name__=='__main__':
    random_seed.set_seed(56)
       #initialize checkpoints and loggers
    args = get_args()
    main(args)
   