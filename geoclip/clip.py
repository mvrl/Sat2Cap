import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection, CLIPVisionConfig
import pytorch_lightning as pl
import imageio

class Clip(pl.LightningModule):
    def __init__(self, args, img_type):
        super().__init__()
        self.args = args
        self.img_type=img_type
        self.imo_crop = 320
        self.vit_map = {'32':'openai/clip-vit-base-patch32', '16':'openai/clip-vit-base-patch16', '14L':'openai/clip-vit-large-patch14'}
        self.vit = self.vit_map[args.vit]
        if self.img_type == 'ground_level':
            print(f'Ground Level Clip instantiated with {self.vit}')
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vit)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.vit).eval()
        elif self.img_type == 'overhead':
            print(f'Overhead Clip instantiated with {self.vit}')
            self.config = CLIPVisionConfig.from_pretrained(self.vit)
            self.config.image_size = self.imo_crop
            self.image_processor = CLIPImageProcessor(do_resize=True,size=336, do_center_crop=True, crop_size=(self.imo_crop,self.imo_crop), padding=True)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.vit, 
                                config=self.config, ignore_mismatched_sizes=True).train()
            
        #print(self.vision_model.config)
        
        

    def forward(self,x):
        if self.img_type=='ground_level':
            processed_image = self.image_processor(x, return_tensors='pt', padding=True).to(self.device)
        elif self.img_type=='overhead':
            processed_image = self.image_processor(x, return_tensors='pt', padding=True, do_resize=True).to(self.device)
            #print(self.processed_image.pixel_values.shape)
        batch_tensors = self.vision_model(**processed_image)
        batch_embeddings = batch_tensors.image_embeds
        if self.args.normalize_embeddings:
            batch_embeddings = batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)
        return batch_embeddings

if __name__ == '__main__':

    clip = CLIP()
    img = imageio.imread('/home/a.dhakal/active/user_a.dhakal/clip_map/images/image_0.png')
    output = clip(img)
    print(output.shape)