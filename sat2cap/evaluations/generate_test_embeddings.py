import numpy as np 
import torch
import webdataset as wds 
import os
import sys
import code
import h5py
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import glob
import PIL
from PIL import ImageFile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py as h5
#code.interact(local=dict(globals(), **locals()))

#local imports
from ..models.geomoco import GeoMoCo 
from ..utils.random_seed import set_seed
from ..utils.preprocess import Preprocess
from ..utils import load_model
from ..data.region_data import RegionDataset
from ..multidata import MultiData as MultiData
from ..multidata_raw import MultiDataRaw
from ..models.geomoco import GeoMoCo

def get_dataloader(val_batch_size, val_path):
    loader_args = Namespace()
    loader_args.val_batch_size = val_batch_size
    loader_args.vali_path = val_path
    dataset = MultiData(loader_args).get_ds('test')
    return dataset

def get_args():
    parser = ArgumentParser(description='arguments for generating test data embeddings')

    parser.add_argument('--ckpt_path', type=str, default='root_path/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt')
    parser.add_argument('--val_path', type=str, default='data_dir/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='root_path/logs/geoclip_embeddings/test_set')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    test_dataset = get_dataloader(args.val_batch_size, args.val_path)
    geoclip = GeoMoCo.load_from_checkpoint(args.ckpt_path)
    clip = load_model.load_clip(args.ckpt_path)
    geoclip_encoder = geoclip.imo_encoder.eval()
    clip_encoder = geoclip.img_encoder.eval()
    
    #freeze all models
    for param in geoclip_encoder.parameters():
        param.requires_grad=False

    for param in clip_encoder.parameters():
        param.requires_grad=False

    output_file = f'{args.output_dir}/test_embeddings.h5'
    valloader = wds.WebLoader(test_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=8)


    with h5py.File(output_file, 'w') as f:
        
        print(f'Creating new h5 file:{output_file}')
        dset_overhead_geoclip_embeddings = f.create_dataset('overhead_geoclip_embeddings', shape=(0,args.embedding_size),maxshape=(100000,512), dtype=np.float32)
        dset_overhead_clip_embeddings = f.create_dataset('overhead_clip_embeddings', shape=(0,args.embedding_size),maxshape=(100000,512), dtype=np.float32)
        dset_ground_clip_embeddings = f.create_dataset('ground_clip_embeddings', shape=(0,args.embedding_size),maxshape=(100000,512), dtype=np.float32)

        dset_location = f.create_dataset('location', shape=(0, 2),maxshape=(100000,2), dtype=np.float32)
        dset_key = f.create_dataset('key', shape=(0 ,1),maxshape=(100000,1), dtype=np.float32)
        
        f.attrs['model_path'] = args.ckpt_path
        f.attrs['data_path'] = args.val_path
        f.attrs['general'] = 'Overhead Embeddings for CLIP and GeoCLIP. Ground Embeddings for CLIP. Location and key information'
    
    
        print('Computing Embeddings')
        for sample in tqdm(valloader):
            img, imo,geo_encode,json,key = sample
            #code.interact(local=dict(globals(), **locals()))
            batch_length = len(img)

            #compute embeddings and location
            _,imo_geoclip_embeddings = geoclip_encoder.forward(imo)
            _,imo_clip_embeddings = clip_encoder.forward(imo)
            _,img_clip_embeddings = clip_encoder.forward(img)
            location = np.array([np.array([js['latitude'],js['longitude']]) for js in json])
            key = np.array(key).reshape((-1,1)).astype(int)

            #adjust the size of the file
            new_size = dset_overhead_geoclip_embeddings.shape[0]+batch_length
            dset_overhead_geoclip_embeddings.resize((new_size, args.embedding_size))
            dset_overhead_clip_embeddings.resize((new_size, args.embedding_size))   
            dset_ground_clip_embeddings.resize((new_size, args.embedding_size))   
            dset_location.resize((new_size, 2))
            dset_key.resize((new_size, 1))

            #add data to file
            dset_overhead_geoclip_embeddings[new_size-batch_length:new_size] = imo_geoclip_embeddings.detach().cpu()
            dset_overhead_clip_embeddings[new_size-batch_length:new_size] = imo_clip_embeddings.detach().cpu()
            dset_ground_clip_embeddings[new_size-batch_length:new_size] = img_clip_embeddings.detach().cpu()
            dset_location[new_size-batch_length:new_size] = location
            dset_key[new_size-batch_length:new_size] = key

    print(f'File saved to {output_path}')


    
