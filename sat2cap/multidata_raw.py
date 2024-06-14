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
#local imports
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


### MultiDataRaw returns raw ground_level, overhead image and metadata without any transformations
class MultiDataRaw(object):

    def __init__(self,args):
        self.img_size = 224
        self.args = args

        self.img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(size=(224,224)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

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

        #random dates and time

    def get_ds(self,mode):
        print(f'\nInitializing {mode} dataset')
        if mode=='train':
            self.dataset = wds.WebDataset(self.args.train_path, resampled=True)
            self.dataset = self.dataset.shuffle(1000).decode("pil", handler=wds.warn_and_continue).to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").batched(self.args.train_batch_size).with_epoch(self.args.train_epoch_length)
        
        elif mode=='test':
            self.dataset = wds.WebDataset(self.args.vali_path)
            self.dataset = self.dataset.decode("pil", handler=wds.warn_and_continue).to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").batched(self.args.val_batch_size)
        
        elif mode=='queue':
            self.dataset = wds.WebDataset(self.args.fill_path)
            self.dataset = self.dataset.decode("pil", handler=wds.warn_and_continue).to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").batched(self.args.train_batch_size)
        
        return self.dataset

 

