import os
from PIL import Image 
from imageio import imread
import code
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import random
from argparse import ArgumentParser
#pytorch imports
import torch
import torch.nn as nn
import pytorch_lightning as pl 
from torchvision.datasets import ImageFolder
import torchvision
#huggingface import
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


##local imports
from ..geoclip import GeoClip

class ZeroShot(nn.Module):
    def __init__(self, queries, ckpt_path):
        super().__init__()
        self.queries = queries
        self.ckpt_path = ckpt_path
        if ckpt_path=='baseline':
            self.baseline_vit = 'openai/clip-vit-base-patch32'
            print('Running off the shelf CLIP model')
            self.base_model, self.base_processor = self.get_baseline_clip()
        else:
            self.baseline_vit = 'openai/clip-vit-base-patch32'
            print('Running pretrained GeoClip model')
            self.geoclip = self.get_geoclip()
        self.query_embeddings = self.get_query_embeddings()
    
    def forward(self, batch):
        x,y = batch
        if self.ckpt_path == 'baseline':
            processed_image = self.base_processor(list(x), return_tensors='pt', padding=True)
            batch_tensors = self.base_model(**processed_image)
            batch_embeddings = batch_tensors.image_embeds
            normalized_imo_embeddings = batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)
        else:
            normalized_imo_embeddings = self.geoclip.forward(list(x)).detach()
        similarity_matrix = torch.matmul(normalized_imo_embeddings, self.query_embeddings.t())
       # print(f'Similarity Matrix Shape is {similarity_matrix.shape}')
        _, max_indices = torch.max(similarity_matrix, axis=-1)
        return {'pred':max_indices, 'gt':y}

    #return the CLIP embeddinds for the given queries using pretrained CLIP Text Model
    def get_query_embeddings(self):
        text_tokenizer = AutoTokenizer.from_pretrained(self.baseline_vit)
        processed_queries = text_tokenizer(self.queries, padding=True, return_tensors='pt')
        text_model = CLIPTextModelWithProjection.from_pretrained(self.baseline_vit).eval()
        for params in text_model.parameters():
            params.requires_grad = False
        query_outputs = text_model(**processed_queries)
        query_embeddings = query_outputs.text_embeds.detach()
        query_embeddings = query_embeddings/query_embeddings.norm(p=2,dim=-1,keepdim=True)
        return query_embeddings    

    #load the overhead img encoder from a checkpoint path
    def get_geoclip(self):
        #load geoclip model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        hparams['vit'] = '32' ##need to use this as this attribute was not set when training the current checkpoint
        pretrained_weights = pretrained_ckpt['state_dict']
        model = GeoClip(hparams)
        model.load_state_dict(pretrained_weights)
        geoclip = model.imo_encoder.eval()
        #set all requires grad to false
        for params in geoclip.parameters():
            params.requires_grad=False
        return geoclip 

    def get_baseline_clip(self):
        image_processor = CLIPImageProcessor.from_pretrained(self.baseline_vit)
        vision_model = CLIPVisionModelWithProjection.from_pretrained(self.baseline_vit).eval()
        for params in vision_model.parameters():
            params.requires_grad=False
        return vision_model, image_processor

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


if __name__ == '__main__':
    #set tokenizer parallelism to true
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    #params
    set_seed(24)
    dataset_path = '/home/a.dhakal/active/user_a.dhakal/datasets/RESISC45/NWPU-RESISC45'
    #ckpt_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/y8yd24yj/checkpoints/epoch=0-step=1100-top_k_score=0.736.ckpt' ## vit-32 B
    ckpt_path = 'baseline'
    batch_size = 256
    #configure dataset
    dataset = ImageFolder(dataset_path, transform=torchvision.transforms.ToTensor())
    total_length = len(dataset)
    train_len = int(0.9*total_length)
    test_len = total_length-train_len
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=8,
                                                    pin_memory=True, persistent_workers=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=8,
                                                    pin_memory=True, persistent_workers=True, drop_last=True)
    
    #convert class labels to queries/prompts
    label_mapping = dataset.class_to_idx
    prompt = 'satellite imagery of '
    vowels = ['a','e','i','o','u']
    prompted_mapping = {value:prompt+'a '+key.replace('_',' ') if key[0] not in vowels else prompt+'an '+key.replace('_',' ') for key,value in label_mapping.items()} 
    queries = list(prompted_mapping.values())

    #predict zero shot labels
    zs_model = ZeroShot(queries, ckpt_path).eval()
    zs_model = zs_model
    with torch.no_grad():
        accs = []
        for i,batch in tqdm(enumerate(test_dataloader)):
            x,y = batch
            pred_dict = zs_model((x,y))
            pred_dict = pred_dict
            preds = pred_dict['pred']
            gt = pred_dict['gt']
            pred_mask = preds==gt
            acc = pred_mask.sum()/len(pred_mask) 
            accs.append(acc.numpy())
        print(f'Length of dataset is {batch_size*i}')
        mean_acc = np.mean(accs)
        print(f'The mean accuracy for zero shot classification is {mean_acc}')


    #code.interact(local=dict(globals(), **locals()))

