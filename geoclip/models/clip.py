import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection, CLIPVisionConfig
import pytorch_lightning as pl
import imageio
import sys

class Clip(pl.LightningModule):
    def __init__(self, args, img_type):
        super().__init__()
        self.args = args
        self.img_type=img_type
        self.imo_crop = 224
        self.imo_mean = [0.3670, 0.3827, 0.3338]
        self.imo_std = [0.2209, 0.1975, 0.1988]
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vit_map = {'32':'openai/clip-vit-base-patch32', '16':'openai/clip-vit-base-patch16', '14L':'openai/clip-vit-large-patch14'}
        print(f'Args.vit is {args.vit}')
        self.vit = self.vit_map[args.vit]
        if self.img_type == 'ground_level':
            print(f'Ground Level Clip instantiated with {self.vit}')
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vit)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.vit).eval()
        elif self.img_type == 'overhead':
            print(f'Overhead Clip instantiated with {self.vit}')
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vit)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.vit).train()
        #print(self.vision_model.config)

    def forward(self,x):
        if self.img_type=='ground_level':
            processed_image = self.image_processor(x, return_tensors='pt', padding=True).to(self.vision_model.device)
        elif self.img_type=='overhead':
            processed_image = self.image_processor(x, return_tensors='pt', padding=True, do_resize=True, do_normalize=True, image_mean=self.imo_mean, image_std=self.imo_std).to(self.args.device)
            #print(self.processed_image.pixel_values.shape)
        batch_tensors = self.vision_model(**processed_image)
        unnormalized_batch_embeddings = batch_tensors.image_embeds
        normalized_batch_embeddings = unnormalized_batch_embeddings/unnormalized_batch_embeddings.norm(p=2,dim=-1, keepdim=True)
        return normalized_batch_embeddings, unnormalized_batch_embeddings

if __name__ == '__main__':

    clip = CLIP()
    img = imageio.imread('/home/a.dhakal/active/user_a.dhakal/clip_map/images/image_0.png')
    output = clip(img)
    print(output.shape)