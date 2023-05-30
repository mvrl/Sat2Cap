import torchvision.transforms as transforms
import torch
import numpy as np
from datetime import datetime
import time

def _convert_image_to_rgb(image):
    return image.convert("RGB")

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

class Preprocess():
    def __init__(self):
        self.valid_imo_transforms = transforms.Compose([
            transforms.Resize((224,224)),
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

    def preprocess_overhead(self,overhead_images_old):
        overhead_images_new = torch.stack([self.valid_imo_transforms(img) for img in overhead_images_old])
        return overhead_images_new

    def preprocess_ground(self, ground_images_old):
        ground_images_new = torch.stack([self.valid_img_transforms(img) for img in ground_images_old])
        return ground_images_new

    def preprocess_meta(self, json):
        lat = json['latitude']
        long = json['longitude']
        lat_long_encode = torch.tensor([np.sin(np.pi*lat/90), np.cos(np.pi*lat/90), np.sin(np.pi*long/180), np.cos(np.pi*long/180)])

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

        geo_encode = torch.cat([lat_long_encode, date_encode, time_encode]).to(dtype=torch.float32)        

        return geo_encode

    def get_geo_encode(self, lat,long, date_time='2010-02-12 12:00:53.0'):
        geo_json = {'latitude':lat, 'longitude':long, 'date_taken':date_time}
        geo_encoding = self.preprocess_meta(geo_json)
        return geo_encoding


