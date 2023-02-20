import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection
import pytorch_lightning as pl
import imageio

class Clip(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self,x):
        processed_image = self.image_processor(list(x), return_tensors='pt', padding=True).to(self.device)
        batch_tensors = self.vision_model.forward(**processed_image)
        batch_embeddings = batch_tensors.image_embeds
        if self.args.normalize_embeddings:
            batch_embeddings = batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)
        return batch_embeddings

if __name__ == '__main__':
    clip = CLIP()
    img = imageio.imread('/home/a.dhakal/active/user_a.dhakal/clip_map/images/image_0.png')
    output = clip(img)
    print(output.shape)