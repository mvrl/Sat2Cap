import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import torch
import webdataset as wds
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter
import code
import os
import sys
import numpy as np
import random

#local imports
from .models.geomoco import GeoMoCo
from .models.geoclip import GeoClip



def get_shards(data_path='/home/a.dhakal/active/datasets/YFCC100m/webdataset'):
    all_shards = os.listdir(data_path)
    test_shards = ['9f248448-1d13-43cb-a336-a7d92bc5359e.tar','206faf6d-e5f4-428e-b27c-4a55746d5629.tar']
    fill_shards = ['1ae82f9a-5cf7-4430-bb9b-00f41d2bc9c3.tar']
    
    test_shards = [os.path.join(data_path, shard) for shard in test_shards]
    fill_shards = [os.path.join(data_path, shard) for shard in fill_shards]

    all_shards = [os.path.join(data_path,shard) for shard in all_shards]
    train_shards = [x for x in all_shards if (x not in test_shards) and (x not in fill_shards)]
    return(train_shards, test_shards, fill_shards)

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
    parser.add_argument('--max_steps', type=int, default=35752)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--train_epoch_length', type=int, default=10000)
    parser.add_argument('--val_epoch_length', type=int, default=10)
    parser.add_argument('--val_check_interval', type=int, default=100)

    #wds hparams
    parser.add_argument('--train_batch_size',type=int, default=512)
    parser.add_argument('--val_batch_size', type=int, default=512)

    #cilp specific hparams
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--temp_clip', type=int, default=100)
    parser.add_argument('--vit', type=str, default='32')
    parser.add_argument('--prompt_loss', action='store_true', default=False)

    #optim params
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_its', type=int, default=9000)

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
    parser.add_argument('--wandb_resume', type=str, default='false')
    
    #moco hparams
    parser.add_argument('--moco', action='store_true', default=False)
    parser.add_argument('--queue_size', type=int, default=2560)
    parser.add_argument('--dim_size', type=int, default=512)

    #geoencodings
    parser.add_argument('--geo_encode', action='store_true', default=False)
    parser.add_argument('dropout_rate', type=float, default=0)

    #metrics hparams
    parser.add_argument('--top_k', type=int, default=5)

    #environmet hparams
    parser.add_argument('--precision', type=str, default='highest')

    #inference mode
    parser.add_argument('--inference', type=bool, default=False)


    args = parser.parse_args()
    return args

def main(args):
    #set learning rate logger
    print('Starting Training')
    
    if args.precision:
        print(f'Setting precision to {args.precision}')
        torch.set_float32_matmul_precision(args.precision)
    #initliaze model
    if args.moco:
        geoclip = GeoMoCo(args)
    else:
        geoclip = GeoClip(args)
    
    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    if args.wandb_resume.lower()=='none':
        wb_logger = WandbLogger(save_dir=args.log_dir,project=args.project_name, name=args.run_name, mode=args.wandb_mode)
    else:
        wb_logger = WandbLogger(save_dir=args.log_dir,project=args.project_name, mode=args.wandb_mode, resume=args.wandb_resume)
    #, resume=args.wandb_resume 
    ckpt_monitors = ((
            ModelCheckpoint(monitor='val_loss', filename='{step}-{val_loss:.3f}', mode='min', save_top_k=20, save_last=True),
                ModelCheckpoint(monitor='top_k_score',filename='{epoch}-{step}-{top_k_score:.3f}', mode='max', save_top_k=2, save_last=True)

        ))

    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = pl.Trainer(fast_dev_run=15, max_epochs=4, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=0,
        accelerator=args.accelerator, devices=args.devices, callbacks=[*ckpt_monitors, lr_logger])
    elif args.mode == 'train':
        print('Training Run')
        trainer = pl.Trainer(precision='32', max_steps=args.max_steps, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=0, 
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
            pretrained_dict = checkpoint['state_dict']
            remove_keys = ['queue', 'queue_ptr']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in remove_keys}
            missing_keys, unexpected_keys = geoclip.load_state_dict(pretrained_dict, strict=False)
            print(f'Missing Keys: {missing_keys}\nUnexpected_keys: {unexpected_keys}')
            #trainer.global_step=checkpoint['global_step']
            #trainer.current_epoch=checkpoint['epoch']
            trainer.fit(geoclip)
            #code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    set_seed(56)
    os.environ['TOKENIZERS_PARALLELISM']='true'
    args = get_args()
    train_shards, test_shards, fill_shards = get_shards()
    #set path hyper parameters
    args.train_path = train_shards #os.path.join(args.data_path, '9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    args.vali_path = test_shards
    args.fill_path = fill_shards
    args.test_path = None
    #code.interact(local=dict(globals(), **locals()))
    main(args)


    #code.interact(local=dict(globals(), **locals()))