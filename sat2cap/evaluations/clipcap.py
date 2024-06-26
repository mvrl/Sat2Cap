import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
import code
from argparse import ArgumentParser
import PIL
from ..models.geomoco import GeoMoCo
from ..utils.preprocess import Preprocess


N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

#@title Caption prediction

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def path_to_lat(path):
    img_name = path.split('/')[-1]
    splits = img_name.split('_')
    lat = splits[3]
    long = splits[4].replace('.jpg', "")
    return float(lat), float(long)

def path_to_dt(path, args):
 
    img_name = path.split('/')[-1]
    splits = img_name.split('_')
    date_time = splits[2]
    date,time = date_time.split(' ')

    if not args.time and not args.date:
        return date_time

    if args.time:
        time = f'{args.time}:00:00.0'
        new_date_time = f'{date} {time}'    

    if args.date:
        date = f'2012-{args.date}-01'
        new_date_time = f'{date} {time}'


    print(time)
    return new_date_time



def get_geo_encode(lat,long, date_time='2010-05-05 01:01:53.0'):
    preprocessor = Preprocess()
    geo_json = {'latitude':lat, 'longitude':long, 'date_taken':date_time}
    geo_encoding = preprocessor.preprocess_meta(geo_json)
    return geo_encoding

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='geoclip')
    parser.add_argument('--use_geo', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='root_path/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt')
   # parser.add_argument('--ckpt_path', type=str, default='root_path/logs/GeoClip/f1dtv48z/checkpoints/step=86750-val_loss=4.100.ckpt')
    parser.add_argument('--img_path', type=str, default='root_path/logs/evaluations/wacv/overhead_images/1046311_18_2004-10-24 13:09:39.0_51.818219_4.671249.jpg')
    #parser.add_argument('--date_time', type=str, default='2012-08-20 08:00:00.0')
    parser.add_argument('--time', type=str, default=None)
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--normalization_type', type=str, default='new')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    image_model = args.model_type
    
    use_geo = args.use_geo
    
    #code.interact(local=dict(globals(), **locals()))
    date_time = path_to_dt(args.img_path, args)
    print(date_time)
    #ckpt_path='root_path/logs/GeoClip/u3oyk5ft/checkpoints/step=8600-val_loss=5.672.ckpt'
    #ckpt_path='root_path/logs/temp_models/f1dtv48z/checkpoints/step=38750-val_loss=4.976.ckpt'
    #ckpt_path = 'root_path/logs/GeoClip/s212e5he/checkpoints/step=35750-val_loss=4.972.ckpt' the best one so far
    #ckpt_path ='root_path/logs/temp_models/s212e5he/checkpoints/epoch=3-step=29500-top_k_score=0.920.ckpt'
    #ckpt_path = 'root_path/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt'
    ckpt_path=args.ckpt_path #second best currently using
    #ckpt_path='root_path/logs/GeoClip/r5tztaac/checkpoints/step=6000-val_loss=6.466.ckpt'
    img_path =args.img_path

    #college = 'root_path/logs/evaluations/wacv/test_images/overhead/172549520_18_51.376448_-2.329659.jpg'
    #beach(seagulls)='root_path/logs/evaluations/wacv/test_images/overhead/247264250_18_57.574548_-4.091806.jpg'
    #beach_w_rich_neighborhood = 'root_path/logs/evaluations/wacv/test_images/overhead/354187045_18_36.594994_-4.5195.jpg'
    #japan_disney_world='root_path/logs/evaluations/wacv/test_images/overhead/539178994_18_35.633302_139.882049.jpg'
    #sea facing apartment='root_path/logs/evaluations/wacv/test_images/overhead/1072838342_18_40.773001_9.67947.jpg'
    #
    
    pretrained_model = 'Conceptual captions'  # @param ['COCO', 'Conceptual captions']
    #clip cap model path
    model_path = 'root_path/pretrained_models/clipcap/conceptual_weights.pt'

    is_gpu = True

    device = CUDA(0) if is_gpu else "cpu"
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_length = 10

    model = ClipCaptionModel(prefix_length)

    model.load_state_dict(torch.load(model_path, map_location=CPU)) 

    model = model.eval() 
    device = CUDA(0) if is_gpu else "cpu"
    model = model.to(device)

    #Inference 
    #@title Inference
    use_beam_search = True #@param {type:"boolean"}  

    # image = io.imread(UPLOADED_FILE)
    # pil_image = PIL.Image.fromarray(image)
    #pil_img = Image(filename=UPLOADED_FILE)
    pil_image = PIL.Image.open(img_path)
    #code.interact(local=dict(globals(), **locals()))
    geo_processor = Preprocess()
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        if image_model == 'clip':
            print('Using Regular CLIP')
            clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
            image = preprocess(pil_image).unsqueeze(0).to(device)
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            norm_prefix = prefix/prefix.norm(p=2,dim=-1)
            prefixes = [norm_prefix, prefix]
        elif image_model == 'geoclip':
            print('Using GeoClip')
            checkpoint = torch.load(ckpt_path)
            hparams = checkpoint['hyper_parameters']
            hparams['geo_encode'] = True
            hparams['inference'] = True
            geoclip_model = GeoMoCo(hparams=hparams).eval().to(device)
            unused_params = geoclip_model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f'Couldn\'t load {unused_params}')

            imo_encoder = geoclip_model.imo_encoder
            image = geo_processor.preprocess_overhead([pil_image])

            if use_geo:
                geo_encoder = geoclip_model.geo_encoder.to('cpu')
                lat, long = path_to_lat(img_path)
                if args.normalization_type=='new':
                    geo_encoding = get_geo_encode(lat, long, date_time)
                #  code.interact(local=dict(globals(), **locals()))
                    geo_embeddings = geo_encoder(geo_encoding).to(device)
                    _, unnormalized_imo_embeddings = imo_encoder(image)
                    #unnormalized_imo_embeddings = unnormalized_imo_embeddings.to(device)
                    unnormalized_imo_embeddings = unnormalized_imo_embeddings.to(device)
                    unnormalized_imo_embeddings = unnormalized_imo_embeddings+geo_embeddings 
                    normalized_imo_embeddings = unnormalized_imo_embeddings/unnormalized_imo_embeddings.norm(p=2, dim=-1, keepdim=True)
                    
                    prefixes = [normalized_imo_embeddings, unnormalized_imo_embeddings]
                else:
                    geo_encoding = get_geo_encode(lat, long, date_time)
                #  code.interact(local=dict(globals(), **locals()))
                    geo_embeddings = [geo_encoder(geo_encoding).to(device), geo_encoder(geo_encoding).to(device)]
                    imo_embeddings = [embeddings.to(device) for embeddings in imo_encoder(image)]
                    prefixes = [(im_emb+geo_emb) for (im_emb,geo_emb) in zip(imo_embeddings, geo_embeddings)]

            else:
                prefixes = [embeddings.to(device) for embeddings in imo_encoder(image)]
        
        code.interact(local=dict(globals(), **locals()))
        prefix_embeds = [model.clip_project(prefix).reshape(1, prefix_length, -1) for prefix in prefixes]
        
    if use_beam_search:
        generated_text_prefix = [generate_beam(model, tokenizer, embed=prefix_embed)[0] for prefix_embed in prefix_embeds]
    else:
        generated_text_prefix = [generate2(model, tokenizer, embed=prefix_embed) for prefix_embed in prefix_embeds]
    
    print(f'\nNormalized: {generated_text_prefix[0]}')
    print(f'\nUnnormalized: {generated_text_prefix[1]}')
