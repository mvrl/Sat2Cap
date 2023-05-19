import numpy as np
import torch
import torch.nn as nn
import clip
import pytorch_lightning as pl
import imageio.v2 as imageio
import sys
import torchvision.transforms as transforms

from argparse import Namespace
import webdataset as wds

#local import
from ..multidata import MultiData


class Clip(pl.LightningModule):
    def __init__(self, args, img_type):
        super().__init__()
        self.args = args
        self.img_type=img_type
        self.imo_crop = 224
        
        #overhead image stats
        self.imo_mean = [0.3670, 0.3827, 0.3338]
        self.imo_std = [0.2209, 0.1975, 0.1988]

        #transforms
        self.img_transforms = transforms.Compose([
            transforms.CenterCrop((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.imo_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.Normalize(mean=[0.3670, 0.3827, 0.3338], std=[0.2209, 0.1975, 0.1988])
        ])

        #VIT setup
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vit_map = {'32':'ViT-B/32', '16':'ViT-B/16'}
        print(f'Args.vit is {args.vit}')
        self.vit = self.vit_map[args.vit]
        
        if self.img_type == 'ground_level':
            print(f'Ground Level Clip instantiated with {self.vit}')
            self.vision_model, self.image_processor = clip.load(self.vit)
            self.vision_model.eval()
        elif self.img_type == 'overhead':
            print(f'Overhead Clip instantiated with {self.vit}')
            self.vision_model, self.image_processor = clip.load(self.vit)
            self.vision_model.train()
        #print(self.vision_model.config)

    def forward(self,x):
        if self.img_type=='ground_level':
            processed_image = self.img_transforms(x).to(self.args.device)
        elif self.img_type=='overhead':
            processed_image = self.imo_transforms(x).to(self.args.device)
            #print(self.processed_image.pixel_values.shape)
        unnormalized_batch_embeddings = self.vision_model.encode_image(processed_image)
        normalized_batch_embeddings = unnormalized_batch_embeddings/unnormalized_batch_embeddings.norm(p=2,dim=-1, keepdim=True)
        return normalized_batch_embeddings, unnormalized_batch_embeddings

if __name__ == '__main__':

    args = Namespace()
    args.vit = '32'
    args.device = 'cuda'
    clip = Clip(args,'overhead')
    #img = imageio.imread('/home/a.dhakal/active/user_a.dhakal/geoclip/images/ground_img.png')
    img = torch.randn(256, 3, 800, 800)

    wds_path = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar'
    args = {'vali_path':wds_path, 'val_batch_size':25, 'train_epoch_length':10, 'normalize_embeddings':True}

    args = Namespace(**args)
    dataset = MultiData(args).get_ds('test')
    
    sample = next(iter(dataset))
    img, imo, geo_encode, json, key = sample
    import code; code.interact(local=dict(globals(), **locals()))
    output = clip(img)
    print(output.shape)

    dataloader = wds.WebLoader(dataset, num_workers=60, batch_size=None)
    for i,sample in enumerate(dataloader):
        img, imo, geo_encode, json, key = sample
        #output=clip(img)
        print(img.shape)
        if i==30:
            break
