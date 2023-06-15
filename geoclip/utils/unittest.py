from ..data import landuse
from ..utils import utils
from argparse import Namespace
from ..multidata_raw import MultiDataRaw
import code


wds_path = '/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar'
#wds_path = '/scratch1/fs1/jacobsn/a.dhakal/yfc100m/93b7d2ae-0c93-4465-bff8-40e719544440.tar'
args = {'vali_path':wds_path, 'val_batch_size':32, 'train_epoch_length':10, 'normalize_embeddings':True}

args = Namespace(**args)
dataset = MultiDataRaw(args).get_ds('test')
sample = next(iter(dataset))
img, imo, meta, key = sample

code.interact(local=dict(globals(), **locals()))
