import numpy as np
from ..multidata import MultiData
from argparse import Namespace 
import webdataset as wds
import code
import torch
from tqdm import tqdm

if __name__ == '__main__':
    input_file = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/c4ce82ef-a7dd-499e-8421-82d2e781cd07.tar'
    batch_size=512
    
    total_sum = torch.tensor([0.0, 0.0, 0.0])
    total_sum_square = torch.tensor([0.0, 0.0, 0.0])
    total_images = 0

    dataset = wds.WebDataset(input_file, resampled=False)
    dataset = dataset.decode('torchrgb').to_tuple("overhead.jpg", "metadata.json").batched(batch_size)
    dataloader = wds.WebLoader(dataset, num_workers=8, batch_size=None)
    for i,data in tqdm(enumerate(dataloader)):
        inputs,_ = data
        total_sum += inputs.sum(axis = [0, 2, 3])
        total_sum_square += (inputs ** 2).sum(axis = [0, 2, 3])
        total_images += len(inputs)
    
    print(f'Total images was {total_images}')

    #compute mean and std
    count = total_images * 800 * 800
    total_mean = total_sum/count
    total_var  = (total_sum_square / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)    

    output_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/geoclip_embeddings/data_params.txt'
    print(f'Mean: {total_mean}\nStd: {total_std}')
    with open(output_path, 'w') as f:
        f.write(f'Mean: {total_mean}\nStd: {total_std}')

    