
# This script is used to compute the Sat2CLIP/CLIP embeddings for a given model and save it 
# as an h5 file. You have option to use or not use geoencodings during inference. The `compute_ground`
# flag computes the CLIP embeddings of the ground level images instead of the overhead images. The
# `normal_clip` flag can be set to True to compute the CLIP embeddings for overhead images.



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
    loader_args.spherical_harmonics = False
    dataset = MultiData(loader_args).get_ds('test')
    return dataset

def get_args():
    parser = ArgumentParser(description='arguments for generating test data embeddings')

    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt')
    parser.add_argument('--val_path', type=str, default='/home/a.dhakal/active/datasets/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    parser.add_argument('--val_batch_size', type=int, default=250)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/cvpr/test_embeddings')
    parser.add_argument('--normal_clip', action='store_true', default=False)
    parser.add_argument('--use_geoencode', action='store_true', default=False)
    parser.add_argument('--output_name', type=str, default='geoclip_w_dropout_train_only')
    parser.add_argument('--compute_ground', action='store_true', default=False)
    parser.add_argument('--info', type=str, default='GeoCLIP embeddings. Dropout used in training and metadata not used in inference')
    parser.add_argument('--max_size', type=int, default=50000)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    test_dataset = get_dataloader(args.val_batch_size, args.val_path)
    data_shape = args.max_size
    ckpt = torch.load(args.ckpt_path)
    hparams = ckpt['hyper_parameters']
    hparams['inference'] = True
    hparams['dropout_rate'] = 0
    hparams['geo_encode'] = True
    hparams['spherical_harmonics']=False
    
    geoclip_model = GeoMoCo(hparams=hparams).eval().to('cuda')
    for param in geoclip_model.parameters():
        param.requires_grad = False
    
    if args.compute_ground:
        print('Running CLIP for ground images')
        clip_encoder = geoclip_model.img_encoder
        args.use_geoencode=False
    
    else:
        if args.normal_clip:
            print('Running normal CLIP')
            geoclip_encoder = geoclip_model.imo_encoder
            args.use_geoencode = False
        
        else:
            print('Running Sat2Clip')
            unused_params = geoclip_model.load_state_dict(ckpt['state_dict'], strict=False)
            print(f'Couldn\'t load {unused_params}')
            geoclip_encoder = geoclip_model.imo_encoder
            if args.use_geoencode:
                print('Using geoencode')
                geo_encoder = geoclip_model.geo_encoder
    

    output_file = f'{args.output_dir}/{args.output_name}.h5'
    valloader = wds.WebLoader(test_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=8)


    with h5py.File(output_file, 'w') as f:

        print(f'Creating new h5 file:{output_file}')
        dset_overhead_geoclip_embeddings = f.create_dataset('overhead_geoclip_embeddings', shape=(data_shape,args.embedding_size), dtype=np.float32)

        dset_location = f.create_dataset('location', shape=(data_shape, 2), dtype=np.float32)
        dset_key = f.create_dataset('key', shape=(data_shape,1), dtype=np.float32)
        
        f.attrs['model_path'] = args.ckpt_path
        f.attrs['data_path'] = args.val_path
        f.attrs['general'] = args.info
    
    
        print('Computing Embeddings')
        curr = 0
        for sample in tqdm(valloader):
            img, imo,geo_encode,json,key = sample
            geo_encode = geo_encode.to('cuda')
            #code.interact(local=dict(globals(), **locals()))
            batch_length = len(img)

            #compute embeddings and location
            if args.compute_ground:
                _, imo_geoclip_embeddings = clip_encoder.forward(img)
            else:
                _,imo_geoclip_embeddings = geoclip_encoder.forward(imo)
                if args.use_geoencode:
                    geo_embeddings = geo_encoder(geo_encode)
                    imo_geoclip_embeddings = imo_geoclip_embeddings + geo_embeddings
            

            location = np.array([np.array([js['latitude'],js['longitude']]) for js in json])
            key = np.array(key).reshape((-1,1)).astype(int)

            #add data to file
            dset_overhead_geoclip_embeddings[curr:curr+args.val_batch_size] = imo_geoclip_embeddings.detach().cpu()
            dset_location[curr:curr+args.val_batch_size] = location
            dset_key[curr:curr+args.val_batch_size] = key

            curr += args.val_batch_size
            if curr >= data_shape:
                print('Max size reached')
                break
                
    print(f'File saved to {output_file}')


    
#test code
# import h5py
# file_1 = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/cvpr/test_embeddings/geoclip_no_dropout_no_meta_in_inference.h5'
# f1 = h5py.File(file_1, 'r')
# loc_1 = f1['location']

# file_2 = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/cvpr/test_embeddings/geoclip_yes_dropout_yes_meta_in_inference.h5'
# f2 = h5py.File(file_2, 'r')
# loc_2 = f2['location']