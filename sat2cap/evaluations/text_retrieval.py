import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter, Namespace 
import code
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModelWithProjection
#local imports
#from ..multidata_2 import MultiData
from ..multidata import MultiData as MultiData
from ..multidata_raw import MultiDataRaw
from ..models.geomoco import GeoMoCo
from ..utils import utils, preprocess
from .metrics import Retrieval


class TextSim():
    def __init__(self,vit_model='openai/clip-vit-base-patch32'):
        self.vit_model = vit_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.vit_model)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(self.vit_model)
    
    def get_text_embed(self, query):
        processed_query = self.tokenizer(query, padding=True, return_tensors='pt')
        text_outputs = self.text_model(**processed_query)
        text_embeddings = text_outputs.text_embeds.detach()
        return text_embeddings


def get_args():
    parser = ArgumentParser(description='arguments for runnning retrieval metrics', formatter_class=RawTextHelpFormatter)

    #parser.add_argument('--ckpt_path', type=str, default='root_path/logs/GeoClip/u3oyk5ft/checkpoints/step=8600-val_loss=5.672.ckpt')
    #parser.add_argument('--ckpt_path', type=str, default='root_path/logs/temp_models/f1dtv48z/checkpoints/step=71250-val_loss=4.357.ckpt')
    parser.add_argument('--ckpt_path', type=str, default='root_path/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt')
    parser.add_argument('--test_path', type=str, default='data_dir/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_topk', action='store_true', default=False)
    parser.add_argument('--k', type=list, default=[5,10])
    #parser.add_argument('--clip', action='store_true', help='Run inference on normal CLIP model')
    parser.add_argument('--input_prompt', type=str, default='a photo of people playing soccer')
    parser.add_argument('--geo_encode', action='store_true', default=False)
    parser.add_argument('--date_time', type=str, default='')
    parser.add_argument('--save_path', type=str, default='root_path/logs/evaluations/wacv/text_retrieval')
    parser.add_argument('--use_clip', type=bool, default=False)
    #parser.add_argument('--g2o', action='store_true')
    args = parser.parse_args()
    return args


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def get_similarity(normalized_overhead_img_embeddings,normalized_ground_img_embeddings):
    return  torch.matmul(normalized_overhead_img_embeddings,normalized_ground_img_embeddings.t()) # (BxB - ground_imgs X overhead_imgs)

def de_norm(x,normalization):
    mean, std = normalization
    z = x * torch.tensor(std).view(3, 1, 1)
    z = z + torch.tensor(mean).view(3, 1, 1)
    return z

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def save_top_k(model, batch,args, k=5):
    geo_processor = preprocess.Preprocess()
    data_size = len(batch)

    ground_images_raw, overhead_images_raw, metadata, keys = batch
    
    overhead_images = geo_processor.preprocess_overhead(overhead_images_raw)
    ground_images = geo_processor.preprocess_ground(ground_images_raw)
    
    
    if not args.date_time:
        print('Using Ground Date_Time')
        geoencode = utils.get_stacked_geoencode(metadata)
    else:
        print('Using Custom Date-Time')
        new_meta = []
        for meta in metadata:
            temp_dict = {}
            temp_dict['latitude'] = meta['latitude']
            temp_dict['longitude'] = meta['longitude']
            temp_dict['date_taken'] = args.date_time
            new_meta.append(temp_dict)
        geoencode = utils.get_stacked_geoencode(new_meta)

    batch = (ground_images.to(args.device), overhead_images.to(args.device), geoencode.to(args.device), metadata, keys)
        
    with torch.no_grad():
        model = model.to(args.device)
        embeddings = model(batch)
    print('Output keys:',embeddings.keys())

    # Define the sample data
    ground_images_raw = [im.resize((224,224)) for im in ground_images_raw]
    sample = {'overhead_images': overhead_images_raw, 'ground_images': ground_images_raw, 'metadata': metadata}


    # Define a dictionary to map the sample ID to an integer index
    sample_index = {num: i for i, num in enumerate(embeddings['keys'])}

    # Convert the embeddings to PyTorch tensors
    overhead_embs = embeddings['overhead_img_embeddings']
#    ground_embs = embeddings['ground_img_embeddings']

    #generate the text embeddings 
    textsim = TextSim()
    query = [args.input_prompt]
    text_embeddings = textsim.get_text_embed(query)

    #normalize the text embeddings
    normalized_text_embeddings = text_embeddings/text_embeddings.norm(p=2,dim=-1,keepdim=True)
    # Compute the pairwise similarity between overhead and ground embeddings
    similarity_matrix = get_similarity(normalized_text_embeddings,overhead_embs.detach())

     # Get the indices of the top k values
    topk_indices = torch.topk(similarity_matrix, k).indices

    # Convert the indices to a list
    topk_indices_list = topk_indices.squeeze().tolist()


    # Extract the images with the highest similarity
    top_images = [overhead_images_raw[idx] for idx in topk_indices_list]
    # Define the desired padding size (in pixels)
    padding = 5

    # Set the padding color
    padding_color = (255, 255, 255)  # White color
    
    padded_imgs = [ImageOps.expand(original_image, border=padding, fill=padding_color) for original_image in top_images]
    
    overhead_img_grid = image_grid(padded_imgs, 2,2)
            
    r = np.random.randint(0,10000)
    overhead_img_grid.save(f'{args.save_path}/{args.use_clip}_{args.input_prompt}.jpg')
        #ground_img_grid.save(f'root_path/logs/evaluations/wacv/retrieval_images/ground_lat_{curr_lat}_long_{curr_long}_{r}.jpg')
        # for j, idx in enumerate(top_ground_indices):
        #     ground_img = sample['ground_images'][idx]
        #     ground_emb = embeddings['ground_img_embeddings'][idx].to('cpu').numpy()
        #     similarity = top_ground_sims[j]
        #     # Save the ground image
        #     # ...
        #     print(f"  {j+1}. Similarity: {similarity:.4f}")

def get_dataloader_1(val_batch_size, vali_path):
    loader_args = Namespace()
    loader_args.val_batch_size = val_batch_size
    loader_args.vali_path = vali_path
    dataset = MultiData(loader_args).get_ds('test')
    return dataset

def get_dataloader_raw(val_batch_size, vali_path):
    loader_args = Namespace()
    loader_args.val_batch_size = val_batch_size
    loader_args.vali_path = vali_path
    dataset = MultiDataRaw(loader_args).get_ds('test')
    return dataset

if __name__ == '__main__':
    utils.set_seed(56)
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
            hparams['prompt_loss'] = None         

            if not args.geo_encode:
                hparams['geo_encode'] = False
            print(f'Geo Encoding Used:{hparams["geo_encode"]}')      
            geoclip = GeoMoCo(hparams=hparams).eval().to(device)

            #set requires grad to false
            for param in geoclip.parameters():
                param.requires_grad=False

            print('Using GeoClip')

            if not args.use_clip:
                print('Using normal CLIP')
                unused_params = geoclip.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f'Unused params {unused_params}')
            else:
                print('Using GeoCLIP')
            
            val_dataloader = get_dataloader_raw(args.batch_size, args.test_path)
            sample = next(iter(val_dataloader))
            print('Samples Loaded')
            print('Saving topk')
            save_top_k(geoclip, sample,args, 4)
    # code.interact(local=dict(globals(), **locals()))
