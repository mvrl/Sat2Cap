
import numpy as np
import torch
import random
import os
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