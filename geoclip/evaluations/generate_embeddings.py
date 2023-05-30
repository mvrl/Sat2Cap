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
#code.interact(local=dict(globals(), **locals()))

#local imports
from ..geomoco import GeoMoCo 
from ..utils.random_seed import set_seed
from ..multidata import MultiData

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/st07vzqb/checkpoints/epoch=0-step=2500-top_k_score=0.820.ckpt')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/geoclip_embeddings')
    parser.add_argument('--test_dir', type=str, default='/home/a.dhakal/active/datasets/YFCC100m/webdataset/*.tar')
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--max_size', default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    set_seed(56)
    #configure params
    args = get_args()
    ckpt_path=args.ckpt_path
    batch_size=args.batch_size
    output_dir=args.output_dir

    #find path of embeddings that already exist
    existing_paths = glob.glob(f'{output_dir}/*.h5')
    existing_files = [path.split('/')[-1].split('.')[0] for path in existing_paths]
    
    #find path to all input shards
    test_dir=args.test_dir
    test_paths = glob.glob(test_dir)
    
    #remove paths which have already been traversed over
    print(f'Total Input Paths: {len(test_paths)}')
    [test_paths.remove(path) if path.split('/')[-1].split('.')[0] in existing_files else 1 for path in test_paths]
    print(f'Remaining Input Paths: {len(test_paths)}')
    embedding_size=args.embedding_size
    max_size = args.max_size
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load pretrained model
    pretrained_model = GeoMoCo.load_from_checkpoint(ckpt_path).eval()
    geoclip = pretrained_model.imo_encoder.eval().to(device)
    for params in geoclip.parameters():
        params.requires_grad=False
    
    for i,test_path in enumerate(test_paths):
        #load the data
        print('______________________________________________________')
        print(f'Running File {i+1}')
        args = {'val_batch_size':batch_size, 'vali_path':test_path}
        args = Namespace(**args)
        dataset = MultiData(args).get_ds('test')
        input_file_name = test_path.split('/')[-1].split('.')[0]
        output_path=f'{output_dir}/{input_file_name}.h5'
        
        
        #initialize the h5 file
        with h5py.File(output_path, 'w') as f:
            
            print(f'Creating new h5 file:{output_path}')
            dset_tensor = f.create_dataset('tensor', shape=(0,embedding_size), maxshape=(max_size, embedding_size), dtype=np.float32)
            dset_location = f.create_dataset('location', shape=(0, 2), maxshape=(max_size, 2), dtype=np.float32)
        
        with h5py.File(output_path, 'a') as f:
            print('Adding data to h5 file')
            print(f'Running on {device}')
            dset_tensor = f['tensor']
            dset_location = f['location']
            for i,sample in tqdm(enumerate(dataset)):
                _,imo,json,_ = sample
                if len(imo)<batch_size:
                    this_bs = len(imo)
                    location = np.array([np.array([js['latitude'],js['longitude']]) for js in json])
                    normalized_imo_embeddings = geoclip.forward(imo).detach()
                    
                    #resize dataset by adding the last batch_size 
                    new_size = dset_tensor.shape[0]+this_bs
                    dset_tensor.resize((new_size,embedding_size))
                    dset_location.resize((new_size,2))

                    #write current batch to dataset
                    dset_tensor[new_size-this_bs:new_size] = normalized_imo_embeddings.cpu()
                    dset_location[new_size-this_bs:new_size] = location

                    print('End of dataset reached\nExiting')
                    continue
                if i*batch_size == max_size:
                    print(f'Max size {max_size} reached\nExiting')
                    break           
                location = np.array([np.array([js['latitude'],js['longitude']]) for js in json])
                normalized_imo_embeddings = geoclip.forward(imo).detach()
                #resize dataset to fit the new embeddings
                new_size = dset_tensor.shape[0]+batch_size
                dset_tensor.resize((new_size,embedding_size))
                dset_location.resize((new_size,2))

                #write data to the dataset
                dset_tensor[new_size-batch_size:new_size] = normalized_imo_embeddings.cpu()
                dset_location[new_size-batch_size:new_size] = location 

            print(f'Number of data points is {i*batch_size}')
            print(f'File saved in {output_path}')

    # f = h5py.File(output_path, 'r') 
    # dset_tensor = f['tensor']
    # dset_location = f['location']
    # print(dset_tensor.shape, dset_location.shape)
    # f.close()



