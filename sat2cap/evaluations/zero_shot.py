import os
from PIL import Image 
from imageio import imread
import code
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import random
from argparse import ArgumentParser, RawTextHelpFormatter
import ssl
#pytorch imports
import torch
import torch.nn as nn
import pytorch_lightning as pl 
from torchvision.datasets import ImageFolder
import torchvision
import torchgeo
#huggingface import
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import rasterio

##local imports
from ..models. geoclip import GeoClip

class ZeroShot(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.hparams = hparams

        self.vit_map = {'32':'openai/clip-vit-base-patch32', '16':'openai/clip-vit-base-patch16', '14L':'openai/clip-vit-large-patch14'}
        self.baseline_vit = self.vit_map[self.hparams.baseline_vit]
        
        if self.hparams.do_interpolate:
            self.geoclip = self.get_interpolate()
        else:
            self.geoclip = self.get_geoclip()
        self.query_embeddings = self.get_query_embeddings().to('cuda')
    
    def forward(self, batch):
        x,y = batch

        normalized_imo_embeddings = self.geoclip.forward(list(x)).detach()
        similarity_matrix = torch.matmul(normalized_imo_embeddings, self.query_embeddings.t())
       # print(f'Similarity Matrix Shape is {similarity_matrix.shape}')
        _, max_indices = torch.max(similarity_matrix, axis=-1)
        return {'pred':max_indices, 'gt':y}

    #return the CLIP embeddinds for the given queries using pretrained CLIP Text Model
    def get_query_embeddings(self):
        text_tokenizer = AutoTokenizer.from_pretrained(self.baseline_vit)
        processed_queries = text_tokenizer(self.hparams.queries, padding=True, return_tensors='pt')
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
        pretrained_ckpt = torch.load(self.hparams.ckpt_path)
        pretrained_hparams = pretrained_ckpt['hyper_parameters']
        pretrained_hparams['prompt_loss']=None
        model = GeoClip(pretrained_hparams)
        if self.hparams.do_baseline:
            geoclip = model.imo_encoder.eval()
        
        else:
            pretrained_weights = pretrained_ckpt['state_dict']
            model.load_state_dict(pretrained_weights)
            geoclip = model.imo_encoder.eval()
        
        #set all requires grad to false
        for params in geoclip.parameters():
            params.requires_grad=False
        return geoclip 

    def get_interpolate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running Interpolated Model')
        pretrained_ckpt = torch.load(self.hparams.ckpt_path)
        pretrained_hparams = pretrained_ckpt['hyper_parameters']
        pretrained_weights = pretrained_ckpt['state_dict']
        pretrained_hparams['prompt_loss']=None
        model_baseline = GeoClip(pretrained_hparams)
        model_trained = GeoClip(pretrained_hparams)
        missing_keys, unexpected_keys = model_trained.load_state_dict(pretrained_weights, strict=False)
        print(f'Missing Keys: {missing_keys}\nUnexpected_keys: {unexpected_keys}')
        
        baseline_geoclip = model_baseline.imo_encoder.eval()
        trained_geoclip = model_trained.imo_encoder.eval()

        for params in baseline_geoclip.parameters():
            params.requires_grad=False

        for params in trained_geoclip.parameters():
            params.requires_grad=False

        baseline_dict = baseline_geoclip.state_dict()
        trained_dict = trained_geoclip.state_dict()

        for key in baseline_dict:
            trained_dict[key] = (1-args.geoclip_wt)*baseline_dict[key]+args.geoclip_wt*trained_dict[key]

        interpolated_model = GeoClip(pretrained_hparams)
        interpolated_geoclip = interpolated_model.imo_encoder.eval()
        
        for params in interpolated_geoclip.parameters():
            params.requires_grad=False
        
        interpolated_geoclip.load_state_dict(trained_dict)
        interpolated_geoclip = interpolated_geoclip.to(device)
        
        return interpolated_geoclip


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

def tif_to_pil(tif):
    return rasterio.open(tif)


def get_args():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    
    #hparams
    parser.add_argument('--num_workers', type=int, default=8)
    #parser.add_argument('--dataset_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/datasets/RESISC45/NWPU-RESISC45')
    parser.add_argument('--batch_size', type=str, default=64)
    parser.add_argument('--train_size', type=float, default=0.5)
    parser.add_argument('--ckpt_path', type=str, default='root_path/logs/GeoClip/st07vzqb/checkpoints/epoch=0-step=2500-top_k_score=0.820.ckpt')
    parser.add_argument('--do_baseline', action='store_true', default=False)
    parser.add_argument('--do_interpolate', action='store_true', default=False)
    parser.add_argument('--geoclip_wt', type=float, default=0.5)
    parser.add_argument('--baseline_vit', type=str, default='32')
    parser.add_argument('--dataset_name', type=str, default='resisc')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #fixes ssl certificate error
    ssl._create_default_https_context = ssl._create_unverified_context
    #set tokenizer parallelism to true
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    #params
    set_seed(24)
    args = get_args()
    
    #configure dataset
    if args.dataset_name.lower()=='resisc':
        print('Eval on RESISC45 dataset')
        args.dataset_path = '/home/a.dhakal/active/user_a.dhakal/datasets/RESISC45/NWPU-RESISC45'
        dataset = ImageFolder(args.dataset_path, transform=torchvision.transforms.ToTensor())
    elif args.dataset_name.lower()=='eurosat':
        print('Eval on EuroSat dataset')
        args.dataset_path = '/home/a.dhakal/active/user_a.dhakal/datasets/eurosat/2750'
        dataset = ImageFolder(args.dataset_path, transform=torchvision.transforms.ToTensor())
    elif args.dataset_name.lower()=='ucmer':
        print('Eval on UCMER dataset')
        args.dataset_path = '/home/a.dhakal/active/user_a.dhakal/datasets/UCMerced_LandUse/Images'
        dataset = ImageFolder(args.dataset_path, transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((256,256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                    ]))
    else:
        raise ValueError('Invalid Dataset Name')
    
    total_length = len(dataset)
    train_len = int(args.train_size*total_length)
    test_len = total_length-train_len
    test_dataset, train_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8,
                                                    pin_memory=True, persistent_workers=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8,
                                                    pin_memory=True, persistent_workers=True, drop_last=True)
    
    #convert class labels to queries/prompts
    label_mapping = dataset.class_to_idx
    prompt = 'satellite imagery of '
    vowels = ['a','e','i','o','u']
    prompted_mapping = {value:prompt+'a '+key.replace('_',' ') if key[0] not in vowels else prompt+'an '+key.replace('_',' ') for key,value in label_mapping.items()} 
    queries = list(prompted_mapping.values())
    args.queries = queries

    evaluation_dir = 'root_path/logs/evaluations/zero_shot_results.txt'
    #alphas = [i/10 for i in range(0,11)]
    alphas=[1]           
    with open(evaluation_dir, 'a') as f:
        to_write = f'alpha\tmean_accuracy\tLength of Test Set\n'
        f.write(to_write)
        f.write('_______________________________________________________\n')
    #predict zero shot labels
    
    for alpha in alphas:
        args.geoclip_wt = alpha
        print(f'Using alpha {args.geoclip_wt}')
        zs_model = ZeroShot(args).eval()
        zs_model = zs_model
        with torch.no_grad():
            accs = []
            for i,batch in tqdm(enumerate(train_dataloader)):
                x,y = batch
                pred_dict = zs_model((x,y))
                pred_dict = pred_dict
                preds = pred_dict['pred'].to('cpu')
                gt = pred_dict['gt'].to('cpu')
                pred_mask = preds==gt
                acc = pred_mask.sum()/len(pred_mask) 
                accs.append(acc.numpy())
            print(f'Length of dataset is {args.batch_size*i}')
            mean_acc = np.mean(accs)
            print(f'The mean accuracy for zero shot classification with aplha {args.geoclip_wt} is {mean_acc}')
            with open(evaluation_dir, 'a') as f:
                to_write = f'{args.geoclip_wt}\t{mean_acc}\t{args.batch_size*i}\n'
                f.write(to_write)
                f.write('_______________________________________________________\n')

    #code.interact(local=dict(globals(), **locals()))

