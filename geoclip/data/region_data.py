import torch
from torch.utils.data import Dataset, DataLoader
import PIL
import numpy as np


from ..utils import preprocess


def get_lat_lon(path):

    img_name = path.split('/')[-1]
    splits = img_name.split('_')
    lat = float(splits[0])
    lon = float(splits[1].replace('.jpg',''))
    return np.array([lat,lon])


class RegionDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.geo_processor = preprocess.Preprocess()
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image_loc = get_lat_lon(image_path)
    
        #preprocess image
        image = self.load_image(image_path)
        
        return image, image_loc 

    def load_image(self, image_path):
        img = PIL.Image.open(image_path)
        processed_image = self.geo_processor.preprocess_overhead([img])[0]
        return processed_image