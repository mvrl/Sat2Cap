import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter 
import code

#local imports
from ..geoclip import GeoClip
from .metrics import Retrieval


def get_args():
    parser = ArgumentParser(description='arguments for runnning retrieval metrics', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/st07vzqb/checkpoints/epoch=0-step=2500-top_k_score=0.820.ckpt')
    parser.add_argument('--test_path', type=str, default='/home/a.dhakal/active/datasets/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--geoclip_wt', type=float, default=0.5)
    args = parser.parse_args()
    return args


def get_retrieval_metric(model, sample, k=1):
    data_size = len(sample)
    embeddings = model(sample)
    print('Output keys:',embeddings.keys())
    ground_img_embeddings = embeddings['ground_img_embeddings']
    overhead_img_embeddings = embeddings['overhead_img_embeddings']
    print(f'Size of retrieval data {len(ground_img_embeddings)}')
    keys = embeddings['keys']
    retrieval = Retrieval(k=k)
    metric = retrieval.fit_k_similar(overhead_img_embeddings, ground_img_embeddings)
    return metric

if __name__ == '__main__':
    args = get_args() 
    device = torch.device(args.device)
    #no gradient context manager for evaluation
    with torch.set_grad_enabled(False):
        #load pretrained weights
            checkpoint = torch.load(args.ckpt_path)
            hparams = checkpoint['hyper_parameters']
    
            #set new hyper parameters
            hparams['val_batch_size'] = args.batch_size
            hparams['test_path'] = args.test_path               
            geoclip = GeoClip(hparams=hparams).eval().to(device)

            #set requires grad to false
            for param in geoclip.parameters():
                param.requires_grad=False
            geoclip.load_state_dict(checkpoint['state_dict'])

            #fetch the test dataloader
            val_dataloader = geoclip.val_dataloader()
            sample = next(iter(val_dataloader))

            #compute the metric for baseline CLIP
            baseline_clip = GeoClip(hparams=hparams).eval().to(device)
            for param in baseline_clip.parameters():
                param.requires_grad=False

            
            #get the interpolated model
            baseline_dict = baseline_clip.state_dict()
            trained_dict = geoclip.state_dict()

            for key in baseline_dict:
                trained_dict[key] = (1-args.geoclip_wt)*baseline_dict[key]+args.geoclip_wt*trained_dict[key]

            interpolated_model = GeoClip(hparams=hparams).eval().to(device)
            interpolated_model.load_state_dict(trained_dict)
            for param in interpolated_model.parameters():
                param.requires_grad=False
        
            geoclip_metric = get_retrieval_metric(geoclip, sample, 5)
            print(f'The retrieval metric for geoclip is {geoclip_metric}')
            baseline_metric = get_retrieval_metric(baseline_clip, sample, 5)
            print(f'The retrieval metric for baseline is {baseline_metric}')
            interpolated_metric = get_retrieval_metric(interpolated_model, sample, 5)
            print(f'The retrieval metric for interpolated model is {interpolated_metric}')
            
# code.interact(local=dict(globals(), **locals()))
