import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection, CLIPVisionConfig
from torchvision.transforms import RandAugment
import pytorch_lightning as pl
import imageio
import sys
from argparse import Namespace
import code

class Clip(pl.LightningModule):
    def __init__(self, args, img_type):
        super().__init__()
        self.args = args
        self.img_type=img_type
        self.imo_crop = 224
        
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vit_map = {'32':'openai/clip-vit-base-patch32', '16':'openai/clip-vit-base-patch16', '14L':'openai/clip-vit-large-patch14'}
        print(f'Args.vit is {args.vit}')
        self.vit = self.vit_map[args.vit]
        if self.img_type == 'ground_level':
            print(f'Ground Level Clip instantiated with {self.vit}')
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.vit).eval()
        elif self.img_type == 'overhead':
            print(f'Overhead Clip instantiated with {self.vit}')
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.vit).train()
        print(f'The device is {self.vision_model.device}')
        #print(self.vision_model.config)

    def forward(self,x):
        x = x.to(self.vision_model.device, dtype=torch.float32)
        batch_tensors = self.vision_model(x)
        unnormalized_batch_embeddings = batch_tensors.image_embeds
        normalized_batch_embeddings = unnormalized_batch_embeddings/unnormalized_batch_embeddings.norm(p=2,dim=-1, keepdim=True)
        return normalized_batch_embeddings, unnormalized_batch_embeddings

if __name__ == '__main__':

    args = Namespace()
    args.vit = '32'
    args.device = 'cuda'
    clip = Clip(args, 'overhead')
    img = torch.randn(256, 3, 224, 224)
    output = clip(img)
    print(output.shape)