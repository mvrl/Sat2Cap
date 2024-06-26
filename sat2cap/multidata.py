import numpy as np
import webdataset as wds
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import scipy
from PIL import Image
import code
from argparse import Namespace
import io
from datetime import datetime
import random
import time
from tqdm import tqdm #local imports
from .models.clip import Clip
from .models.geoencode import GeoNet

def get_random_date(start, end):
    year = random.randint(start, end)
    month = random.randint(1,12)
    day = random.randint(1, 30)
    date = f"{year}-{month:02}-{day:02}"
    return date

def get_random_time(start,end):
    random_hr = random.randint(7,23)
    time_str = f'{random_hr}:00:00.0'
    return time_str

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class MultiData(object):

    def __init__(self,args):
        self.img_size = 224
        self.args = args

        #set transforms for GL image
        self.img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(size=(224,224)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        #set transforms for overhead image
        self.imo_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandAugment(num_ops=3, interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988))
        ])

        self.imo_transforms_original = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(size=(224,224)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988))
        ])

        # set transforms for validation
        self.valid_img_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(size=(224,224)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        self.valid_imo_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(size=(224,224)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988))
        ])

    #initialize the webdataset
    def get_ds(self,mode):
        print(f'\nInitializing {mode} dataset')
        if mode=='train':
            self.dataset = wds.WebDataset(self.args.train_path, resampled=True)
            self.dataset = self.dataset.shuffle(1000).decode("pil", handler=wds.warn_and_continue).to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").map(self.do_transforms, handler=wds.warn_and_continue).batched(self.args.train_batch_size).with_epoch(self.args.train_epoch_length)
        
        elif mode=='test':
            self.dataset = wds.WebDataset(self.args.vali_path)
            self.dataset = self.dataset.decode("pil", handler=wds.warn_and_continue).to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").map(self.do_valid_transforms, handler=wds.warn_and_continue).batched(self.args.val_batch_size)
        
        elif mode=='queue':
            self.dataset = wds.WebDataset(self.args.fill_path)
            self.dataset = self.dataset.decode("pil", handler=wds.warn_and_continue).to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").map(self.do_valid_transforms, handler=wds.warn_and_continue).batched(self.args.train_batch_size)
        
        return self.dataset

    def do_valid_transforms(self, sample):
        img, imo,json, key = sample
        img = self.valid_img_transforms(img)
        imo = self.valid_imo_transforms(imo)
        lat = json['latitude']
        long = json['longitude']

        date_time = json['date_taken']
        try:
            date_str = date_time.split(' ')[0]
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        except (IndexError, ValueError) as e:
            date_str = get_random_date(2000,2015)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        try:
            time_str = date_time.split(' ')[1]
            time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
        except (IndexError, ValueError) as e:
            time_str = get_random_time(7,23)
            time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')

        
        #extract data info
        max_year = 2015
        min_year = 2000
        year = date_obj.year
        year = (year - max_year)/(max_year - min_year)
        month = date_obj.month
        day = date_obj.day

        #extract time info
        hour = time_obj.hour
        
        #date encoding
        date_encode = torch.tensor([np.sin(2*np.pi*year), np.cos(2*np.pi*year),np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), np.sin(2*np.pi*day/31), np.cos(2*np.pi*day/31)])

        #time encoding
        time_encode = torch.tensor([np.sin(2*np.pi*hour/23), np.cos(2*np.pi*hour/23)])


        if not self.args.spherical_harmonics:    
            lat_long_encode = torch.tensor([np.sin(np.pi*lat/90), np.cos(np.pi*lat/90), np.sin(np.pi*long/180), np.cos(np.pi*long/180)])
            geo_encode = torch.cat([lat_long_encode, date_encode, time_encode]).to(dtype=torch.float32)
        else:
            geo_encode = torch.cat([torch.tensor([long]), torch.tensor([lat]), date_encode, time_encode]).to(dtype=torch.float32)
        return img, imo,geo_encode,json,key

    def do_transforms(self, sample):
        img, imo,json, key = sample
        img = self.img_transforms(img)
        imo = self.imo_transforms(imo)
        lat = json['latitude']
        long = json['longitude']
        

        date_time = json['date_taken']
        #incase date_time corrupt, select a random date
        try:
            date_str = date_time.split(' ')[0]
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        except (IndexError, ValueError) as e:
            date_str = get_random_date(2000,2015)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        try:
            time_str = date_time.split(' ')[1]
            time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
        except (IndexError, ValueError) as e:
            time_str = get_random_time(7,23)
            time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')

        
        #extract data info
        max_year = 2015
        min_year = 2000
        year = date_obj.year
        year = (year - max_year)/(max_year - min_year)
        month = date_obj.month
        day = date_obj.day

        #extract time info
        hour = time_obj.hour
        
        #date encoding
        date_encode = torch.tensor([np.sin(2*np.pi*year), np.cos(2*np.pi*year),np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), np.sin(2*np.pi*day/31), np.cos(2*np.pi*day/31)])

        #time encoding
        time_encode = torch.tensor([np.sin(2*np.pi*hour/23), np.cos(2*np.pi*hour/23)])

        #use spherical harmonics encoding.
        if not self.args.spherical_harmonics:
            lat_long_encode = torch.tensor([np.sin(np.pi*lat/90), np.cos(np.pi*lat/90), np.sin(np.pi*long/180), np.cos(np.pi*long/180)])
            geo_encode = torch.cat([lat_long_encode, date_encode, time_encode]).to(dtype=torch.float32)
        else:
            geo_encode = torch.cat([torch.tensor([long]), torch.tensor([lat]), date_encode, time_encode]).to(dtype=torch.float32)
        
        return img, imo,geo_encode,json,key


