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
from .evaluations.metrics import Retrieval
import os
import random


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

#computer cross entropy for the similarity matrix both rowwise and columnwise
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
        self.test_path = self.hparams.test_path

        #this is deprecated in favor of self.save_hyperparameters(hparams)
        #self.hparams = hparams
        #todo:modify the img_processor config file for the large overhead images
        #self.imo_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')

        #frozen image encoder to get ground level image CLIP embeddings
        self.img_encoder = Clip(self.hparams,'ground_level')
    
        if self.hparams.freeze_clip:
            self.img_encoder.eval()
            for params in self.img_encoder.parameters():
                params.requires_grad=False
      
        #trainable image encoder to get ground level image CLIP embeddings
        self.imo_encoder = Clip(self.hparams, 'overhead')
        self.imo_encoder.train()

        #instantiate the learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/self.hparams.temperature))
        self.temp_clip = self.hparams.temp_clip
        
        #check if a valid data path is provided
        if self.train_path:
            self.trainset = MultiData(self.hparams).get_ds(mode='train')
        else:
            raise ValueError('Valid path to webdataset file is required')
        
        #test for validation file
        if self.vali_path:
            self.valiset = MultiData(self.hparams).get_ds(mode='test')
        else:
            self.valiset = None

        #test for test file
        if self.test_path:
            self.testset = MultiData(self.hparams).get_ds(mode='test')
        else:
            self.testset = None

        print(f'Ground Level Image encoder training mode:{self.img_encoder.training}')
        print(f'Overhead Image encoder training model:{self.imo_encoder.training}')

    #forward function that runs during inference
    def forward(self, batch):
        img, imo, _,keys = batch
        #get the ground img encodings and detach
        with torch.set_grad_enabled(False): #equivalent to torch.no_grad()
            ground_img_embeddings = self.img_encoder(img).to(self.device)
        #get the overhead image embeddings undetached
        overhead_embeddings = self.imo_encoder(imo).to(self.device)
        
        return{'ground_img_embeddings':ground_img_embeddings, 
            'overhead_img_embeddings':overhead_embeddings,
            'keys':keys
        }
    
    #shred step for test and validation that returns embeddings and clip loss
    def shared_step(self,batch):
        #get embeddings from forward function
        embeddings = self(batch)
        normalized_ground_img_embeddings = embeddings['ground_img_embeddings']
        normalized_overhead_img_embeddings = embeddings['overhead_img_embeddings']
        
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
    
    #forward pass for each batch in training 
    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        loss = outputs['loss']
        self.log('loss', loss, sync_dist=True, batch_size=self.hparams.train_batch_size)
        return loss 
    
    #forward pass for each batch in validation
    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        val_loss = outputs['loss']
        self.log('val_loss', val_loss, sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
        return {'val_loss':outputs['loss'],'normalized_ground_img_embeddings':outputs['normalized_ground_img_embeddings'], 'normalized_overhead_img_embeddings':outputs['normalized_overhead_img_embeddings']}

    #compute retrieval metrics for a random batch of validation 
    def validation_epoch_end(self, outputs):
        if len(outputs)==0:
            print('Skipping Validatiion Epoch End')
            pass
        else:
            random_batch = np.random.randint(0,len(outputs))
            validation_embeddings = outputs[random_batch]
            ground_img_embeddings = validation_embeddings['normalized_ground_img_embeddings']
            overhead_img_embeddings = validation_embeddings['normalized_overhead_img_embeddings']
            retrieval = Retrieval(k=self.hparams.top_k)
            retrieval_metric = retrieval.fit_k_similar(overhead_img_embeddings, ground_img_embeddings)
            self.log(f'top_k_score', retrieval_metric, sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
            #print(f'Retrieval Metric on validation set is {retrieval_metric}') 
            return {'top_k_score':retrieval_metric}
          
    def train_dataloader(self):

        trainloader = wds.WebLoader(self.trainset, batch_size=None,
                    shuffle=False, pin_memory=True, num_workers=self.hparams.num_workers)
        trainloader = trainloader.unbatched().shuffle(10000).batched(self.hparams.train_batch_size)
        return trainloader

    def val_dataloader(self):
        if self.valiset:
            valloader = wds.WebLoader(self.valiset, batch_size=None, #batch_size=self.hparams.val_batch_size,
                    shuffle=False, pin_memory=True, num_workers=self.hparams.num_workers)
            valloader = valloader.unbatched().shuffle(10000).batched(self.hparams.val_batch_size)
            return valloader
        pass

    def test_dataloader(self):
        if self.testset:
            testloader = wds.WebLoader(self.testset, batch_size=None, #batch_size=self.hparams.val_batch_size,
                    shuffle=False, pin_memory=True, num_workers=self.hparams.num_workers)
            #testloader = testloader.unbatched().shuffle(10000).batched(self.hparams.val_batch_size)
            return testloader

    def configure_optimizers(self):
        print(f'Initializing Learning rate {self.hparams.learning_rate}')
        self.optim = torch.optim.AdamW(filter(lambda p:p.requires_grad, self.imo_encoder.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=0.2,
            betas=(0.9,0.98),
            eps=1e-6
        )
        
        self.warm_up_iterations = 2000
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = self.optim,
            T_0 = self.warm_up_iterations
        )
        return {'optimizer': self.optim, 
        'lr_scheduler': {
            'name':'trian/lr',
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }
        }

   # **************For contrastive learning with soft labels****************************
    # def cross_entropy(self, preds, targets, reduction='none'):
    #     log_softmax = nn.LogSoftmax(dim=-1)
    #     loss = (-targets*log_softmax(preds)).sum(1)
    #     if reduction=='none':
    #         return loss
    #     elif reduction == 'mean':
    #         return loss.mean()

def get_shards(data_path='/home/a.dhakal/active/datasets/YFCC100m/webdataset'):
    all_shards = os.listdir(data_path)
    test_shards = ['9f248448-1d13-43cb-a336-a7d92bc5359e.tar','206faf6d-e5f4-428e-b27c-4a55746d5629.tar']
    test_shards = [os.path.join(data_path, shard) for shard in test_shards]
    all_shards = [os.path.join(data_path,shard) for shard in all_shards]
    train_shards = [x for x in all_shards if x not in test_shards]
    return(train_shards, test_shards)

def set_seed(seed: int = 56) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_args():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    #training hparams
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--normalize_embeddings', type=bool, default=True)
    parser.add_argument('--freeze_clip', type=bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--train_epoch_length', type=int, default=10000)
    parser.add_argument('--val_epoch_length', type=int, default=10)
    parser.add_argument('--val_check_interval', type=int, default=100)

    #wds hparams
    parser.add_argument('--train_batch_size',type=int, default=512)
    parser.add_argument('--val_batch_size', type=int, default=700)

    #cilp specific hparams
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--temp_clip', type=int, default=100)
    parser.add_argument('--vit', type=str, default='14')

    #optim params
    parser.add_argument('--learning_rate', type=float, default=5e-5)

    #data hparams
    parser.add_argument('--data_path', type=str, default='/home/a.dhakal/active/datasets/YFCC100m/webdataset')
    parser.add_argument('--input_size', type=int, default=800)

    #logging hparams
    parser.add_argument('--log_dir', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs')
    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/or5comrl/checkpoints/epoch=22-step=16215.ckpt')
    parser.add_argument('--ckpt_mode', type=str, default='hard')
    parser.add_argument('--project_name', type=str, default='GeoClip')
    parser.add_argument('--run_name', type=str, default='geoclip_large')
    parser.add_argument('--wandb_mode', type=str, default='online')
    

    #metrics hparams
    parser.add_argument('--top_k', type=int, default=5)

    args = parser.parse_args()
    return args

def main(args):
    #set learning rate logger
    print('Starting Training')
    
    #initliaze model
    geoclip = GeoClip(args)
    
    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    wb_logger = WandbLogger(save_dir=args.log_dir,project=args.project_name, name=args.run_name, mode=args.wandb_mode) # resume='y8yd24yj'
    ckpt_monitors = ((
            ModelCheckpoint(monitor='val_loss', filename='{step}-{val_loss:.3f}', mode='min', save_top_k=2, save_last=True),
                ModelCheckpoint(monitor='top_k_score',filename='{epoch}-{step}-{top_k_score:.3f}', mode='max', save_top_k=2, save_last=True)

        ))

    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = pl.Trainer(fast_dev_run=6, max_epochs=4, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=0,
        accelerator=args.accelerator, devices=args.devices, callbacks=[*ckpt_monitors, lr_logger])
    elif args.mode == 'train':
        print('Training Run')
        trainer = pl.Trainer(precision=16, max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=0, 
        accelerator=args.accelerator, devices=args.devices, callbacks=[*ckpt_monitors, lr_logger], 
        val_check_interval=args.val_check_interval, check_val_every_n_epoch=None, limit_val_batches=args.val_epoch_length,
        log_every_n_steps=15)
    else:
        raise ValueError('Invalid value for mode')
    
    if args.ckpt_path.lower()=='none'.lower():
        trainer.fit(geoclip)
    else:
        if args.ckpt_mode.lower()=='hard':
            print('Hard Checkpoint Reload')
            trainer.fit(geoclip, ckpt_path=args.ckpt_path)
        elif args.ckpt_mode.lower()=='soft':
            print('Soft Checkpoint Reload')
            checkpoint = torch.load(args.ckpt_path)
            geoclip.load_state_dict(checkpoint['state_dict'])
            #trainer.global_step=checkpoint['global_step']
            #trainer.current_epoch=checkpoint['epoch']
            trainer.fit(geoclip)
            #code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    set_seed(56)
    args = get_args()
    train_shards, test_shards = get_shards()
    #set path hyper parameters
    args.train_path = train_shards #os.path.join(args.data_path, '9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    args.vali_path = test_shards
    args.test_path = None
    main(args)


    #code.interact(local=dict(globals(), **locals()))