import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy
from PIL import Image
import code
from argparse import Namespace
import io

#local imports
from .models.clip import Clip


class MultiData(object):

    def __init__(self,args):
        self.img_size = 224
        self.args = args

    def get_ds(self,mode):
        print(f'\nInitializing {mode} dataset')
        if mode=='train':
            self.dataset = wds.WebDataset(self.args.train_path, resampled=True, handler=wds.warn_and_continue)
            self.dataset = self.dataset.shuffle(1000).decode("pil").to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").map(self.do_transforms).batched(self.args.train_batch_size).with_epoch(self.args.train_epoch_length)
        
        elif mode=='test':
            self.dataset = wds.WebDataset(self.args.vali_path, resampled=False, handler=wds.warn_and_continue)
            self.dataset = self.dataset.decode("pil").to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").map(self.do_transforms).batched(self.args.val_batch_size)
        
        elif mode=='queue':
            self.dataset = wds.WebDataset(self.args.fill_path, resampled=False, handler=wds.warn_and_continue)
            self.dataset = self.dataset.decode("pil").to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").map(self.do_transforms).batched(self.args.train_batch_size)
        
        return self.dataset


    def do_transforms(self, sample):
        img, imo,json, key = sample
        img = img.resize((224,224), resample=Image.BICUBIC)
        return img, imo,json, key

    # def my_jpg_decoder(self, key, value):
    #     if not key.endswith(".jpg"):
    #         return None
    #     try:
    #         with io.BytesIO(value) as stream:
    #             img = Image.open(stream)
    #             img.load()
    #             img = img.convert("RGB")
            
    #     except:
    #         img = None
    #     return img 


if __name__ == '__main__':
    wds_path = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar'
    args = {'vali_path':wds_path, 'val_batch_size':16, 'train_epoch_length':10, 'normalize_embeddings':True}

    args = Namespace(**args)
    dataset = MultiData(args).get_ds('test')
    
    sample = next(iter(dataset))
    img, imo, json, key = sample
    ov1 = imo[0]
    gr1 = img[0]
    #code.interact(local=dict(globals(), **locals()))

    r = np.random.randint(0,40)

    for i, data in enumerate(dataset):
        img, imo, json, key = sample
        n = len(imo)
        for im,js in zip(imo,json):
            curr_lat = js['latitude']
            curr_long = js['longitude']
            im.save(f'/home/a.dhakal/active/user_a.dhakal/geoclip/images/overhead_images/imo_lat_{curr_lat}_long_{curr_long}_{r}.jpg')
            
    # clip_overhead = Clip(args,'overhead')
    # clip_ground = Clip(args,'ground_level')

   
#   
