import numpy as np
import h5py
import code
import matplotlib.pyplot as plt
import glob
import pandas as pd
import torch

#local import
from ..models.geomoco import GeoMoCo 
from ..utils.preprocess import Preprocess
from ..utils.random_seed import set_seed
form ..utils import utils

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/path/to/h5/file')
    parser.add_argument('--batch_size', type=int, default=540)
    parser.add_argument('--output_path', type=str, default='/path/to/csv/file')
    parser.add_argument('--input_prompt', type=str, default='playing in the sand with family')
    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt')
    parser.add_argument('--date_time', type=str, default='2012-05-20 08:00:00.0')

    args = parser.parse_args()
    return args

def get_geo_embeddings(locations, date_time)
    geo_processor = Preprocess()
    all_encodings = []
    for location in locations:
        lat = location[0]
        lon = location[1]
        
        geo_encoding = geo_processor.get_geo_encode(lat, lon, date_time)
        all_encodings.append(geo_encoding)
    all_encodings = torch.stack(all_encodings)
    return all_encodings

    

if __name__ == '__main__':
    set_seed(56)
    args = get_args()
    handle = h5py.File(args.input_file, 'r')
    print(f'The keys are {handle.keys()}')
    
    overhead_embeddings = handle['tensor'][:]
    locations = handle['location'][:]

    print(f'Length of data is {len(locations)}')

    #get the sin-cos encoding for date time and location
    geo_encodings = get_geo_embeddings(locations, args.date_time)

    #get the geo embeddings
    pretrained_model = GeoMoCo.load_from_checkpoint(args.ckpt_path).eval()
    
    geo_encoder = pretrained_model.geo_encoder.to('cuda')
    geo_encodings = geo_encodings.to('cuda')
    geoembeddings = geo_encoder(geo_encodings)

    geoembeddings = geoembeddings.to('cpu')
    unnormalized_embeddings = overhead_embeddings + geoembeddings
    


        