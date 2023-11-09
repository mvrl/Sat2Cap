
# This script allows us to generate text using clip-based embeddings
# You can use a pretrained model and webdataset to generate- NOT PREFERRED SLOW
# The precomputed flag allows you to use precomputed embeddings save in h5 file
# Use the `save_embeddings.py` script to precompute the embeddings for your model


import webdataset as wds
import torch
from argparse import Namespace, ArgumentParser
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import code
from tqdm import tqdm
import csv
import numpy as np
import h5py as h5
import os

from ..multidata import MultiData
from .clipcap import ClipCaptionModel, ClipCaptionPrefix, generate_beam, generate2
from ..models.geomoco import GeoMoCo

def get_model(model_path, prefix_length):
    pretrained_model = 'Conceptial captions'
    is_gpu=True
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path))
    model = model.eval().to('cuda')
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='geoclip')
    parser.add_argument('--use_geo', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt')
   # parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt')
    parser.add_argument('--img_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/overhead_images/1046311_18_2004-10-24 13:09:39.0_51.818219_4.671249.jpg')
    #parser.add_argument('--date_time', type=str, default='2012-08-20 08:00:00.0')
    parser.add_argument('--time', type=str, default=None)
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--normalization_type', type=str, default='new')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_data', type=str, default='/home/a.dhakal/active/datasets/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    parser.add_argument('--clipcap_model_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/pretrained_models/clipcap/conceptual_weights.pt')
    parser.add_argument('--output_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/cvpr/generated_text')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--precomputed', type=bool, default=True)
    parser.add_argument('--embeddings_path', type=str, default='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    prefix_length = 10
    text_model = get_model(args.clipcap_model_path, prefix_length).to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if not args.precomputed:
        print('Using the pretrained model')
        ckpt = torch.load(args.ckpt_path)
        hparams = ckpt['hyper_parameters']
        hparams['geo_encode'] = True
        geoclip_model = GeoMoCo(hparams=hparams).eval().to(args.device)
        for param in geoclip_model.parameters():
            param.requires_grad = False

        normal_clip = geoclip_model.img_encoder.eval().to(args.device)
        for param in normal_clip.parameters():
            param.requires_grad = False

        unused_params = geoclip_model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'Couldn\'t load {unused_params}')


        dataset_args = {'vali_path':args.test_data, 'val_batch_size':args.batch_size}
        dataset_args = Namespace(**dataset_args)

        dataset = MultiData(dataset_args).get_ds('test')
        
        test_loader = wds.WebLoader(dataset, batch_size=None,
                        shuffle=False, pin_memory=True, num_workers=8)
        
        output_path = args.output_path+'/clipcap_generated_text.csv'
        with open(output_path,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Ground Level Text', 'CLIP Overhead Text', 'Sat2Clip Overhead Text'])
            for i, sample in tqdm(enumerate(test_loader)):
                overhead_text = []
                ground_text = []
                overhead_clip_text = []

                sample[0] = sample[0].to('cuda')
                sample[1] = sample[1].to('cuda')
                sample[2] = sample[2].to('cuda')
                embeddings = geoclip_model(sample)
                un_ground_embeds = embeddings['unnormalized_ground_img_embeddings'] .to(args.device)
                un_overhead_embeds = embeddings['unnormalized_overhead_img_embeddings'].to(args.device)
                overhead_clip_embeds = normal_clip(sample[1])[1].to(args.device)

                overhead_prefix_embeds = text_model.clip_project(un_overhead_embeds).reshape(un_ground_embeds.shape[0], prefix_length,-1)
                ground_prefix_embeds = text_model.clip_project(un_ground_embeds).reshape(un_ground_embeds.shape[0], prefix_length,-1)
                clip_overhead_prefix_embeds = text_model.clip_project(overhead_clip_embeds).reshape(un_ground_embeds.shape[0], prefix_length,-1)
            
                for overhead in overhead_prefix_embeds:
                    overhead = torch.unsqueeze(overhead, 0)
                    generated_overhead_text = generate_beam(text_model, tokenizer, embed=overhead, beam_size=1)[0]
                    overhead_text.append(generated_overhead_text)
                for ground in ground_prefix_embeds:
                    ground = torch.unsqueeze(ground, 0)
                    generated_ground_text = generate_beam(text_model, tokenizer, embed=ground, beam_size=1)[0]
                    ground_text.append(generated_ground_text)
                for clip_overhead in clip_overhead_prefix_embeds:
                    clip_overhead = torch.unsqueeze(clip_overhead, 0)
                    generated_clip_overhead_text = generate_beam(text_model, tokenizer, embed=clip_overhead, beam_size=1)[0]
                    overhead_clip_text.append(generated_clip_overhead_text)
                
                to_write = np.array([ground_text, overhead_clip_text, overhead_text]).T
                # code.interact(local=dict(globals(), **locals()))
                writer.writerows(to_write)

    elif args.precomputed:
        print('Using precomputed embeddings')
        fname = (args.embeddings_path.split('/')[-1]).split('.')[0]
        output_path = os.path.join(args.output_path,fname+'.csv')

        handle = h5.File(args.embeddings_path, 'r')
        embeddings_ds = handle['overhead_geoclip_embeddings']
        embeddings = torch.from_numpy(embeddings_ds[:,:]).to('cuda')
        
        embeddings_prefix = text_model.clip_project(embeddings).reshape(embeddings.shape[0], prefix_length, -1)
        all_text = []
        with open(output_path,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i, prefix in tqdm(enumerate(embeddings_prefix)):
                prefix = torch.unsqueeze(prefix,0)
                generated_text = generate_beam(text_model, tokenizer, embed=prefix, beam_size=1)[0]
                all_text.append([generated_text])
            writer.writerows(all_text)
    
    print('All text written to file')
    handle.close()

       
        
       




       
       