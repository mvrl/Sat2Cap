import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy
from PIL import Image
import code
from argparse import Namespace
from .clip import Clip


class MultiData(object):

    def __init__(self,args):
        self.img_size = 224
        self.args = args

    def get_ds(self,mode):
        print(f'Initializing {mode} dataset')
        if mode=='train':
            self.dataset = wds.WebDataset(self.args.train_path, resampled=True)
            self.dataset = self.dataset.shuffle(1000).decode('pil').to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").map(self.do_transforms).batched(self.args.train_batch_size).with_epoch(self.args.train_epoch_length)
        elif mode=='test':
            self.dataset = wds.WebDataset(self.args.vali_path, resampled=True)
            self.dataset = self.dataset.decode('pil').to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__").map(self.do_transforms).batched(self.args.val_batch_size).with_epoch(self.args.val_epoch_length)
        return self.dataset


    def do_transforms(self, sample):
        img, imo, json, key = sample

        img = img.resize((224,224), resample=Image.BICUBIC)
        # self.transforms_img = transforms.Compose([
        #     transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC)
        #     transforsm.CenterCrop(size=(224,224)),
        #     transforms.ToTensor()
        # ])
        
        # self.transforms_imo = transforms.Compose([
        #     transforms.Resize(size=(800,800), interpolation=transforms.InterpolationMode.BICUBIC)
        #     transforms.CenterCrop(size=(800,800)),
        #     transforms.ToTensor()
        # ])

        # img = self.transforms_img(img)
        # imo = self.transforms_imo(imo)
        return img, imo, json, key


if __name__ == '__main__':
    wds_path = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar'
    args = {'train_path':wds_path, 'train_batch_size':16, 'train_epoch_length':10, 'normalize_embeddings':True}

    args = Namespace(**args)
    dataset = MultiData(args).get_ds('train')
    
    sample = next(iter(dataset))
    img, imo, _, _ = sample
    print(len(img), len(imo))

    clip_overhead = Clip(args,'overhead')
    clip_ground = Clip(args,'ground_level')

    code.interact(local=dict(globals(), **locals()))
#   
