from ..multidata_raw import MultiDataRaw
from ..utils.preprocess import Preprocess
from argparse import Namespace
from argparse import ArgumentParser, RawTextHelpFormatter
import torch
import code
from PIL import Image

def get_args():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    #training hparams
    parser.add_argument('--overhead_path', type=str, default='root_path/logs/evaluations/wacv/overhead_images')
    parser.add_argument('--ground_path', type=str, default='root_path/logs/evaluations/wacv/ground_images')

    parser.add_argument('--wds_path', type=str, default='data_dir/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')

    parser.add_argument('--data_size', type=int, default=1000)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    preprocessor = Preprocess()
    args = get_args()
    save_path_imo = args.overhead_path
    save_path_img = args.ground_path

    wds_path = args.wds_path
    #wds_path = '/scratch1/fs1/jacobsn/a.dhakal/yfc100m/93b7d2ae-0c93-4465-bff8-40e719544440.tar'
    hparams = {'vali_path':wds_path, 'val_batch_size':args.data_size, 'train_epoch_length':10, 'normalize_embeddings':True}

    hparams = Namespace(**hparams)
    dataset = MultiDataRaw(hparams).get_ds('test')
    img_old, imo_old, json_old, key_old = next(iter(dataset))
    img_old = [im.resize((224,224), resample=Image.BICUBIC) for im in img_old]
    img = preprocessor.preprocess_ground(img_old) 
    imo = preprocessor.preprocess_overhead(imo_old)
    geoencode = torch.stack([preprocessor.preprocess_meta(js) for js in json_old])
   # code.interact(local=dict(globals(), **locals()))

    for img, imo, json, key in zip(img_old, imo_old, json_old,key_old):
        lat = json['latitude']
        long = json['longitude']
        date_time = json['date_taken']
        img.save(f'{save_path_img}/{key}_{date_time}_{lat}_{long}.jpg')
        imo.save(f'{save_path_imo}/{key}_{date_time}_{lat}_{long}.jpg')


