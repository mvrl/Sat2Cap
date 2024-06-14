import numpy as np 
import torch
import code

#local import
from ..geoclip import GeoClip
if __name__ == '__main__':
    ckpt_path = 'root_path/logs/GeoClip/st07vzqb/checkpoints/epoch=0-step=2500-top_k_score=0.820.ckpt'
    out_path = 'root_path/logs/GeoClip/base_model/CLIP_32B.pt'
    pretrained_ckpt = torch.load(ckpt_path)
    pretrained_hparams = pretrained_ckpt['hyper_parameters']
    #code.interact(local=dict(globals(), **locals()))
    model = GeoClip(pretrained_hparams).eval()
    torch.save({
        'epoch':0,
        'state_dict':model.state_dict(),
        'hyper_parameters':pretrained_hparams,
        'pytorch-lightning_version':'1.9.1',
        'hparams_name':'hparams'
    }, out_path)
    print(f'Model saved to {out_path}')
    