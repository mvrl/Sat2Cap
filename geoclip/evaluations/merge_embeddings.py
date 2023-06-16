import numpy as np
import h5py
import code
import matplotlib.pyplot as plt
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os
#local import
from ..data.region_data import DynamicDataset
from ..models.geomoco import GeoMoCo 
from ..utils.preprocess import Preprocess
from ..utils.random_seed import set_seed
from ..utils import utils

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/geoclip_embeddings/netherlands/no_dropout/step=38000-val_loss=4.957.h5')
    parser.add_argument('--batch_size', type=int, default=20000)
    #parser.add_argument('--input_prompt', type=str, default='playing in the sand with family')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--date_time', type=str, default='2012-05-20 08:00:00.0')

    args = parser.parse_args()
    return args

def get_geo_embeddings(locations, date_time):
    geo_processor = Preprocess()
    all_encodings = []
    for location in locations:
        lat = location[0]
        lon = location[1]
        
        geo_encoding = geo_processor.get_geo_encode(lat, lon, date_time)
        all_encodings.append(geo_encoding)
    all_encodings = torch.stack(all_encodings)
    return all_encodings

def ensure_dir(args):
    #extract directory with overhead embeddings
    input_dir = args.input_path.split('/')[0:-1]
    input_dir = '/'.join(input_dir)

    #extract model name
    model_name = args.input_path.split('/')[-1].split('.')[0]

    #make better date time
    date_time = args.date_time.split('.')[0].replace(' ','_').replace(':','-')

    #make output dir 
    output_dir = f'{input_dir}/{date_time}'
    output_path = f'{output_dir}/dynamic_{model_name}.h5'

    if os.path.exists(output_path):
        raise ValueError('The model already exists')
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    return output_path


if __name__ == '__main__':
    set_seed(56)
    args = get_args()
    handle = h5py.File(args.input_file, 'r')
    print(f'The keys are {handle.keys()}')
    
    overhead_embeddings = handle['overhead_embeddings'][:]
    locations = handle['location'][:]

    print(f'Length of data is {len(locations)}')

    #get ckpt path if not given
    if not args.ckpt_path:
        print('Using ckpt path from h5 file')
        args.ckpt_path = handle.attrs['model_path']
        print(f'Using model {handle.attrs['model_path']}')

    handle.close()

    #get the dynamic dataset
    dynamic_dataset = DynamicDataset(locations, args.date_time)

    #create dataloader
    dynamic_dataloader = DataLoader(dynamic_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    #get the geo encoder
    pretrained_model = GeoMoCo.load_from_checkpoint(args.ckpt_path).eval()
    geo_encoder = pretrained_model.geo_encoder.to('cuda')
    for params in geo_encoder.parameters():
        params.requires_grad=False


    #compute and store all geo embeddings in the all_geo_embeddings
    all_geo_embeddings = torch.empty(0)
    for batch in tqdm(dynamic_dataloader):
        batch = batch.to('cuda')
        geo_embeddings = geo_encoder(batch).detach().cpu()
        all_geo_embeddings = torch.cat([all_geo_embeddings, geo_embeddings])

#    code.interact(local=dict(globals(), **locals()))
    # compute the unnormalized dynamic embeddings
    dynamic_embeddings = all_geo_embeddings + overhead_embeddings

    #store the dynamic embeddings in h5 file
    with h5py.File(output_path, 'w') as f:
        
        print(f'Creating new h5 file:{output_path}')
        dset_tensor = f.create_dataset('tensor', shape=dynamic_embedding.shape, dtype=np.float32)
        dset_location = f.create_dataset('location', shape=locations.shape, dtype=np.float32)
        
        dset_location.attrs['input_file_path'] = args.input_file
        dset_location.attrs['input_region'] = handle['location'] 

    

    


        