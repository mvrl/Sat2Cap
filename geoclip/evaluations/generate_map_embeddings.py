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
#code.interact(local=dict(globals(), **locals()))

#local imports
from ..models.geomoco import GeoMoCo 
from ..utils.random_seed import set_seed
from ..multidata import MultiData
from ..utils.preprocess import Preprocess

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt')
    parser.add_argument('--batch_size', type=int, default=540)
    parser.add_argument('--output_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/geoembed_embeddings/netherlands/no_dropout/step=38000-val_loss=4.957.h5')
    parser.add_argument('--test_dir', type=str, default='/home/a.dhakal/active/proj_smart/BING_IMG/netherland_bing/')

    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--max_size', default=147420)

    args = parser.parse_args()
    return args

def get_lat_lon(img_paths):
    lat_lon = []
    for path in img_paths:
        img_name = path.split('/')[-1]
        splits = img_name.split('_')
        lat = float(splits[0])
        lon = float(splits[1].replace('.jpg',''))
        lat_lon.append(np.array([lat, lon]))
    return np.array(lat_lon)

if __name__ == '__main__':
    set_seed(56)
    #configure params
    args = get_args()
    ckpt_path=args.ckpt_path
    batch_size=args.batch_size
    test_dir = args.test_dir

    #find path of embeddings that already exist
    test_paths = glob.glob(f'{test_dir}/*/*.jpg')
    #existing_files = [path.split('/')[-1].split('.')[0] for path in existing_paths]
    
    #remove paths which have already been traversed over
    print(f'Total Input Images: {len(test_paths)}')
    assert(len(test_paths)==args.max_size)

    embedding_size=args.embedding_size
    max_size = args.max_size
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #initialize the h5 file
    output_path = args.output_path
    with h5py.File(output_path, 'w') as f:
        
        print(f'Creating new h5 file:{output_path}')
        dset_tensor = f.create_dataset('tensor', shape=(0,embedding_size), maxshape=(max_size, embedding_size), dtype=np.float32)
        dset_location = f.create_dataset('location', shape=(0, 2), maxshape=(max_size, 2), dtype=np.float32)
        

    #load pretrained model
    pretrained_model = GeoMoCo.load_from_checkpoint(ckpt_path).eval()
    overhead_encoder = pretrained_model.imo_encoder.eval().to(device)
    for params in geoclip.parameters():
        params.requires_grad=False

    #load the overhead image preprocessor
    geo_processor = Preprocess()
     
     # calculate the number of steps
    num_steps = len(test_paths)/batch_size
    curr_start = 0 #initialize a pointer
    
    with h5py.File(output_path, 'a') as f:
        print('Adding data to h5 file')
        print(f'Running on {device}')
        dset_tensor = f['tensor']
        dset_location = f['location']
        for i in range(num_steps):
            curr_img_paths = test_paths[curr_start:curr_start+batch_size]

            #get lat longs for the current image
            curr_lat_lon = get_lat_lon(curr_img_paths)

            #process the current images
            curr_pil_imgs = [PIL.Image.open(img_path) for img_path in curr_img_paths]
            processed_imgs = geo_processor.preprocess_overhead(curr_pil_imgs)
            
            #get the unnormalized embeddings
            _, unnormalized_batch_embeddings = overhead_encoder.forward(processed_imgs)
            dset_tensor[curr_start:curr_start+batch_size] = unnormalized_batch_embeddings.detach()
            dset_location[curr_start:curr_start+batch_size] = curr_lat_lon


        print(f'Number of data points is {i*batch_size}')
        print(f'File saved in {output_path}')

    # f = h5py.File(output_path, 'r') 
    # dset_tensor = f['tensor']
    # dset_location = f['location']
    # print(dset_tensor.shape, dset_location.shape)
    # f.close()



