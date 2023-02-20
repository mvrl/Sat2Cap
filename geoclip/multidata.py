import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy
from PIL import Image
import code

class MultiData(object):

    def __init__(self, wds_path):
        self.img_size = 224
        self.wds_path = wds_path

    def get_ds(self,mode):
        print(f'Initializing {mode} dataset')
        if mode=='train':
            self.dataset = wds.WebDataset(self.wds_path, resampled=True)
            self.dataset = self.dataset.shuffle(1000).decode('rgb').to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__")
        elif mode=='test':
            self.dataset = wds.WebDataset(self.wds_path)
            self.dataset = self.dataset.decode('rgb').to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json","__key__")
        self.dataset = self.dataset.map(self.do_transforms)
        return self.dataset


    def to_pil(self, batched_img):
        pils = [Image.from_array(img) for img in batched_img]
        return pils

    def do_transforms(self, sample):
        img, imo, json, key = sample
        self.transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BILINEAR)
        ])
        
        self.transforms_imo = transforms.Compose([
            transforms.ToTensor()
        ])

        img = self.transforms_img(img)
        imo = self.transforms_imo(imo)
        return img, imo, json, key

if __name__ == '__main__':
    wds_path = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar'
    dataset = MultiData(wds_path).get_ds()
    #code.interact(local=dict(globals(), **locals()))
    sample = next(iter(dataset))
    img, imo, _, _ = sample
    print(img.shape, imo.shape)

#   
