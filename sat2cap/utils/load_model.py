from ..models.geomoco import GeoMoCo
import torch

def load_geomoco(ckpt_path):
    print('Soft Checkpoint Reload')
    checkpoint = torch.load(ckpt_path)
    hparams = checkpoint['hyper_parameters']
    #if this model does not have the geo_encode settings set it to False
    if 'geo_encode' not in hparams:
        hparams['geo_encode'] = False

    geoclip = GeoMoCo(hparams=hparams)
    print('Using Pretrained GeoClip')
    unused_params = geoclip.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f'Unused params {unused_params}')
    return geoclip

def load_clip(ckpt_path):
    print('Soft Checkpoint Reload')
    checkpoint = torch.load(ckpt_path)
    hparams = checkpoint['hyper_parameters']
    clip = GeoMoCo(hparams=hparams)
    print('Using CLIP')
    return clip  