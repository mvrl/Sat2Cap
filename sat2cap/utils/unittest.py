from ..data import landuse
from ..utils import utils
from argparse import Namespace
from ..multidata_raw import MultiDataRaw
from ..data.region_data import RegionDataset, DynamicDataset
import code
import glob
import numpy as np
from torch.utils.data import DataLoader
import h5py
# wds_path = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar'
# #wds_path = '/scratch1/fs1/jacobsn/a.dhakal/yfc100m/93b7d2ae-0c93-4465-bff8-40e719544440.tar'
# args = {'vali_path':wds_path, 'val_batch_size':32, 'train_epoch_length':10, 'normalize_embeddings':True}

# args = Namespace(**args)
# dataset = MultiDataRaw(args).get_ds('test')
# sample = next(iter(dataset))
# img, imo, meta, key = sample

# date_time = '2012-05-20 08:00:00.0'
# locations = np.array([[49.9475,-5.18],[50.43, -3.47],[60.17,0.789],[30.33,10.39]])
# dataset = DynamicDataset(locations, date_time)
# dataloader = DataLoader(dataset, shuffle=False, batch_size=2, num_workers=0)
# sample = next(iter(dataloader))
# code.interact(local=dict(globals(), **locals()))
output_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/geoclip_embeddings/netherlands/no_dropout/2012-05-20_08-00-00/dynamic_step=38000-val_loss=4.h5'
with h5py.File(output_path,'r') as f:
    print(f.keys())
    l = f['location']
    code.interact(local=dict(globals(), **locals()))