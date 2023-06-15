from ..data import landuse
from ..utils import utils
from argparse import Namespace
from ..multidata_raw import MultiDataRaw
from ..data.region_data import RegionDataset
import code
import glob
from torch.utils.data import DataLoader

# wds_path = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar'
# #wds_path = '/scratch1/fs1/jacobsn/a.dhakal/yfc100m/93b7d2ae-0c93-4465-bff8-40e719544440.tar'
# args = {'vali_path':wds_path, 'val_batch_size':32, 'train_epoch_length':10, 'normalize_embeddings':True}

# args = Namespace(**args)
# dataset = MultiDataRaw(args).get_ds('test')
# sample = next(iter(dataset))
# img, imo, meta, key = sample

img_paths = glob.glob('/home/a.dhakal/active/proj_smart/BING_IMG/netherland_bing/*/*.jpg')
print(f'Num Images: {len(img_paths)}')
dataset = RegionDataset(img_paths)
dataloader = DataLoader(dataset, shuffle=False, batch_size=100, num_workers=0)
sample = next(iter(dataloader))
img, loc = sample
code.interact(local=dict(globals(), **locals()))
