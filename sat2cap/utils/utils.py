
import numpy as np
import torch
import random
import os
import h5py

from .preprocess import Preprocess

preprocessor = Preprocess()
#give lat long from path
def path_to_lat(path):
    img_name = path.split('/')[-1]
    splits = img_name.split('_')
    lat = splits[2]
    long = splits[3].replace('.jpg', "")
    return float(lat), float(long)

#sets random seet
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

def get_geo_encode(lat,long, date_time='2010-05-05 01:00:00.0'):

    geo_json = {'latitude':lat, 'longitude':long, 'date_taken':date_time}
    geo_encoding = preprocessor.preprocess_meta(geo_json)
    return geo_encoding

def get_stacked_geoencode(json):
    geo_encoding = torch.stack([preprocessor.preprocess_meta(js) for js in json])
    return geo_encoding

#returns the average distance between embeddings in a given test set
def get_avg_cosine(input_path):
    with h5py.File(input_path, 'r') as handle:
        print(handle.keys())
        overhead_embeddings = handle['overhead_embeddings'][0:100000]
        normalized_overhead_embeddings = overhead_embeddings/np.linalg.norm(overhead_embeddings, axis=-1, keepdims=True, ord=2)
        sim_matrix = normalized_overhead_embeddings @ normalized_overhead_embeddings.T

    return sim_matrix.mean()

if __name__ == '__main__':

    input_path = 'root_path/logs/geoclip_embeddings/england/clip/step=86750-val_loss=4.h5'
    mean = get_avg_cosine(input_path)
    print(mean)