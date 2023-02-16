import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import torch
import webdataset as wds
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter
from transformers import CLIPImageProcessor, CLIPVisionConfig
import code
#local imports
from .clip import Clip
from .multidata import MultiData
import numpy as np


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    overhead_img_loss = contrastive_loss(similarity)
    ground_img_loss = contrastive_loss(similarity.t())
    return (overhead_img_loss + ground_img_loss) / 2.0

class GeoClip(pl.LightningModule):
    def __init__(self, hparams):
        #save hyperparameters
        super().__init__()
        self.save_hyperparameters(hparams)
        
        #set path attributes
        self.train_path = self.hparams.train_path
        self.vali_path = self.hparams.vali_path

        #this is deprecated in favor of self.save_hyperparameters(hparams)
        #self.hparams = hparams
        #todo:modify the img_processor config file for the large overhead images
        #self.imo_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')

        #frozen image encoder to get ground level image CLIP embeddings
        self.img_encoder = Clip(self.hparams)
        if self.hparams.freeze_clip:
            for params in self.img_encoder.parameters():
                params.requires_grad=False
      
        #trainable image encoder to get ground level image CLIP embeddings
        self.imo_encoder = Clip(self.hparams)

        #instantiate the learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/self.hparams.temperature))
        self.temp_clip = self.hparams.temp_clip
        
        #check if a valid data path is provided
        if self.train_path:
            self.trainset = MultiData(self.train_path).get_ds()
        else:
            raise ValueError('Path to webdataset file is required')
        
        if self.vali_path:
            self.valiset = MultiData(self.vali_path).get_ds()
        else:
            self.valiset = None
    
    def forward(self, batch):
        img, imo, json = batch
        #get the ground img encodings and detach
        with torch.set_grad_enabled(False):
            ground_img_embeddings = self.img_encoder(img).detach().to(self.device)
        #get the overhead image embeddings undetached
        overhead_embeddings = self.imo_encoder(imo).to(self.device)
        
        return{'ground_img_embeddings':ground_img_embeddings, 
            'overhead_embeddings':overhead_embeddings
        }
    
    def shared_step(self,batch):
        #get embeddings from forward function
        embeddings = self(batch)
        normalized_ground_img_embeddings = embeddings['ground_img_embeddings']
        normalized_overhead_img_embeddings = embeddings['overhead_embeddings']
        
        #Calculate loss
        logit_scale = self.logit_scale.exp()
        #clip scale value if greater than 100
        if logit_scale > 100:
            logit_scale = 100
            self.logit_scale = torch.log(self.temp_clip)
        #similarity between the ground level and overhead imagery
        logits_per_overhead_img = torch.matmul(normalized_overhead_img_embeddings,normalized_ground_img_embeddings.t())*logit_scale
        logits_per_ground_img = logits_per_overhead_img.t() 

        #compute cross_entropy loss between the cross-modal similarities and hard gt
        loss = clip_loss(logits_per_overhead_img)

        return{'loss':loss, 
                'logits_per_overhead_img': logits_per_overhead_img,
                'normalized_ground_img_embeddings':normalized_ground_img_embeddings,
                'normalized_overhead_img_embeddings':normalized_overhead_img_embeddings
        }


    #   ***************For soft targets instead of hard ([0,0,1,0,0]) targets.*******************************
   
    #           https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2
        #similarity between different overhead imagery in the batch
        #overhead_similarity = overhead_embeddings @ overhead_embeddings.T
        #similarity between different ground_level imagery in the batch 
        #ground_similarity = ground_img_embeddings @ ground_img_embeddings.T
        # #target simlarity between gronud level and overhead images
        # targets = nn.functional.softmax((ground_similarity+overhead_similarity)/2 * self.hparams.temperature, dim=1)
        # #loss for ground_img i.e. across different rows
        # ground_img_loss = self.cross_entropy(logits, targets, reduction='none')
        # #loss for overhead_img i.e. across different columns
        # overhead_img_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        # loss = (ground_img_loss + overhead_img_loss)/2.0 #[batch_size]
        # loss = loss.mean()
        # return {'loss':loss, 'overhead_loss':overhead_img_loss,'ground_loss':ground_img_loss}

    def training_step(self, batch):
        outputs = self.shared_step(batch)
        loss = outputs['loss']
        self.log('loss', loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss 

    def validation_step(self, batch):
        outputs = self.shared_step(batch)
        val_loss = outputs['loss']
        self.log('val_loss', val_loss, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss 
          
    def train_dataloader(self):
        #self.trainset = self.trainset.batched(self.hparams.batch_size)
        return wds.WebLoader(self.trainset, num_workers=self.hparams.num_workers,
         batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)

    def val_dataloader(self):
        if self.valiset:
            #self.valiset = self.valiset.batched(self.hparams.batch_size)
            return wds.WebLoader(self.valiset, num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True)
        pass

    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.imo_encoder.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=0.2,
            betas=(0.9,0.98),
            eps=1e-6
        )
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = self.optim,
            T_0 = 1000
        )
        return {'optimizer': self.optim, 'lr_scheduler': self.scheduler}

   # **************For contrastive learning with soft labels****************************
    # def cross_entropy(self, preds, targets, reduction='none'):
    #     log_softmax = nn.LogSoftmax(dim=-1)
    #     loss = (-targets*log_softmax(preds)).sum(1)
    #     if reduction=='none':
    #         return loss
    #     elif reduction == 'mean':
    #         return loss.mean()

def get_args():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    #training hparams
    parser.add_argument('--batch_size',type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--normalize_embeddings', type=bool, default=True)
    parser.add_argument('--freeze_clip', type=bool, default=True)

    #cilp specific hparams
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--temp_clip', type=int, default=100)

    #optim params
    parser.add_argument('--learning_rate', type=float, default=5e-5)

    #data hparams
    parser.add_argument('--train_path', type=str, default='/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar')
    parser.add_argument('--vali_path', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=800)

    #logging hparams
    parser.add_argument('--log_dir', type=str, default='/home/a.dhakal/active/user_a.dhakal/clip_map/logs')
    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/clip_map/logs/GeoClip/6fatux01/checkpoints/epoch=7-step=5640.ckpt')
    parser.add_argument('--project_name', type=str, default='GeoClip')
    parser.add_argument('--run_name', type=str, default='geoclip_2')
    
    #trainer hparams
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()
    return args

def main(args):
    #set learning rate logger
    torch.manual_seed(56)
    print('Starting Training')
    
    #check for checkpoint
    ckpt_path = args.ckpt_path
    geoclip = GeoClip(args).eval()
    train_ds = geoclip.train_dataloader()
    
    #######################################################################################
    # sample = next(iter(train_ds))
    # img,imo,_=sample
    # print(img.shape, imo.shape)
    
    # output = geoclip(sample)
    # code.interact(local=dict(globals(), **locals()))
    #######################################################################################
    #checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    wb_logger = WandbLogger(save_dir=args.log_dir,project=args.project_name, name=args.run_name)
    ckpt_monitors = (
            ModelCheckpoint(monitor='loss', mode='min', save_top_k=3),
        )

    if args.mode == 'dev_run':
        trainer = pl.Trainer(limit_train_batches=10, max_epochs=4, logger=wb_logger, strategy=args.strategy, 
        accelerator=args.accelerator, devices=args.devices, callbacks=[*ckpt_monitors, lr_logger])
    else:
        trainer = pl.Trainer(max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, 
        accelerator=args.accelerator, devices=args.devices, callbacks=[*ckpt_monitors, lr_logger])
    
    trainer.fit(geoclip, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    args = get_args()
   # code.interact(local=dict(globals(), **locals()))
    main(args)

