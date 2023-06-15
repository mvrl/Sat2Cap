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
#code.interact(local=dict(globals(), **locals()))

#local imports
from ..models.geomoco import GeoMoCo 
from ..utils.random_seed import set_seed
from ..multidata import MultiData
from ..utils.preprocess import Preprocess
from ..data.region_data import RegionDataset




def get_lat_lon(img_paths):
    lat_lon = []

    for path in img_paths:
        img_name = path.split('/')[-1]
        splits = img_name.split('_')
        lat = float(splits[0])
        lon = float(splits[1].replace('.jpg',''))
        lat_lon.append(np.array([lat, lon]))

    return np.array(lat_lon)




def get_args():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt')
    parser.add_argument('--batch_size', type=int, default=540)
    parser.add_argument('--output_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/geoembed_embeddings/netherlands/no_dropout/step=38000-val_loss=4.957.h5')
    parser.add_argument('--test_dir', type=str, default='/home/a.dhakal/active/proj_smart/BING_IMG/netherland_bing/')

    parser.add_argument('--embedding_size', type=int, default=512)
    #parser.add_argument('--max_size', default=147420)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    set_seed(56)

    #allow loading truncated files
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    #configure params
    args = get_args()
    batch_size=args.batch_size

    #find path of embeddings that already exist
    test_paths = glob.glob(f'{args.test_dir}/*/*.jpg')

    #create a dataset of the images
    dataset = RegionDataset(test_paths)

    #create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    print(f'Total Input Images: {len(dataset)}')
    #max size is the number of test_path
    max_size = len(test_paths)    
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #initialize the h5 file
    
    with h5py.File(args.output_path, 'w') as f:
        
        print(f'Creating new h5 file:{args.output_path}')
        dset_tensor = f.create_dataset('tensor', shape=(0,args.embedding_size), maxshape=(max_size, args.embedding_size), dtype=np.float32)
        dset_location = f.create_dataset('location', shape=(0, 2), maxshape=(max_size, 2), dtype=np.float32)
        

    #load pretrained model
    pretrained_model = GeoMoCo.load_from_checkpoint(args.ckpt_path).eval()
    overhead_encoder = pretrained_model.imo_encoder.eval().to(device)
    for params in overhead_encoder.parameters():
        params.requires_grad=False

    #load the overhead image preprocessor
    geo_processor = Preprocess()
     
     # calculate the number of steps
    num_steps = round(len(dataset)/batch_size)


    
    curr_start = 0 #initialize a pointer
    
    with h5py.File(args.output_path, 'a') as f:
        print('Adding data to h5 file')
        print(f'Running on {device}')
        dset_tensor = f['tensor']
        dset_location = f['location']
        for i,sample in tqdm(enumerate(dataloader)):
            processed_imgs, locations = sample
            if len(locations)==batch_size:
                #get the unnormalized embeddings
                _, unnormalized_batch_embeddings = overhead_encoder.forward(processed_imgs)
                dset_tensor[curr_start:curr_start+batch_size] = unnormalized_batch_embeddings.detach().cpu()
                dset_location[curr_start:curr_start+batch_size] = locations
                
                curr_start = curr_start+batch_size
            else:    
                #get the unnormalized embeddings
                _, unnormalized_batch_embeddings = overhead_encoder.forward(processed_imgs)
                dset_tensor[curr_start:] = unnormalized_batch_embeddings.detach().cpu()
                dset_location[curr_start:] = locations

        print(f'File saved in {args.output_path}')

    # f = h5py.File(output_path, 'r') 
    # dset_tensor = f['tensor']
    # dset_location = f['location']
    # print(dset_tensor.shape, dset_location.shape)
    # f.close()



