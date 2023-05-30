from ..multidata_2 import MultiData
import code
from argparse import Namespace
import torch
import torchvision.transforms as transforms

topil = transforms.ToPILImage(mode='RGB')

img_normalize = [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]]
imo_normalize = [[0.3670, 0.3827, 0.3338], [0.2209, 0.1975, 0.1988]]

# img_denormalize = transforms.Compose([ transforms.Normalize(mean = (0., 0., 0. ),
#                                                      std = (0.26862954, 0.26130258, 0.27577711)),
#                                 transforms.Normalize(mean = (0.48145466, 0.4578275, 0.40821073),
#                                                      std = ( 1., 1., 1. ))
#                                ])

def de_norm(x,normalization):
    mean, std = normalization
    z = x * torch.tensor(std).view(3, 1, 1)
    z = z + torch.tensor(mean).view(3, 1, 1)
    return z

wds_path = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar'
    #wds_path = '/scratch1/fs1/jacobsn/a.dhakal/yfc100m/93b7d2ae-0c93-4465-bff8-40e719544440.tar'
args = {'vali_path':wds_path, 'val_batch_size':256, 'train_epoch_length':10, 'normalize_embeddings':True}

args = Namespace(**args)
dataset = MultiData(args).get_ds('test')

img, imo, geo_encode, json, key = next(iter(dataset))
code.interact(local=dict(globals(), **locals()))
img_raw = [topil(de_norm(im, img_normalize)) for im in img]
imo_raw = [topil(de_norm(im, imo_normalize)) for im in imo]
# imo_raw = [topil() for im in imo]
#code.interact(local=dict(globals(), **locals()))

for i,im in enumerate(imo_raw):
    im.save(f'/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/junk/{i}.jpg')