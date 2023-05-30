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
    lat = splits[2]
    long = splits[3].replace('.jpg', "")
    return float(lat), float(long)

def get_geo_encode(lat,long, date_time='2010-05-12 12:00:53.0'):
    preprocessor = Preprocess()
    geo_json = {'latitude':lat, 'longitude':long, 'date_taken':date_time}
    geo_encoding = preprocessor.preprocess_meta(geo_json)
    return geo_encoding

if __name__ == '__main__':
    image_model = 'clip'
    
    use_geo = True
    date_time = '2012-05-20 23:00:00.0'
    
    #ckpt_path='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/u3oyk5ft/checkpoints/step=8600-val_loss=5.672.ckpt'
    #ckpt_path='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/f1dtv48z/checkpoints/step=38750-val_loss=4.976.ckpt'
    #ckpt_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/s212e5he/checkpoints/step=35750-val_loss=4.972.ckpt' the best one so far
    ckpt_path ='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/temp_models/s212e5he/checkpoints/epoch=3-step=29500-top_k_score=0.920.ckpt'
    img_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/test_images/overhead/247264250_18_57.574548_-4.091806.jpg'


    pretrained_model = 'Conceptual captions'  # @param ['COCO', 'Conceptual captions']
    #clip cap model path
    model_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/pretrained_models/clipcap/conceptual_weights.pt'

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
            prefixes = [prefix, norm_prefix]
        elif image_model == 'geoclip':
            print('Using GeoClip')
            checkpoint = torch.load(ckpt_path)
            hparams = checkpoint['hyper_parameters']
            hparams['geo_encode'] = True
            geoclip_model = GeoMoCo(hparams=hparams).eval().to(device)
            unused_params = geoclip_model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f'Couldn\'t load {unused_params}')

            imo_encoder = geoclip_model.imo_encoder
            image = geo_processor.preprocess_overhead([pil_image])

            if use_geo:
                geo_encoder = geoclip_model.geo_encoder.to('cpu')
                lat, long = path_to_lat(img_path)
                geo_encoding = get_geo_encode(lat, long, date_time)
            #  code.interact(local=dict(globals(), **locals()))
                geo_embeddings = [geo_encoder(geo_encoding).to(device), geo_encoder(geo_encoding).to(device)]
                imo_embeddings = [embeddings.to(device) for embeddings in imo_encoder(image)]
                prefixes = [(im_emb+geo_emb) for (im_emb,geo_emb) in zip(imo_embeddings, geo_embeddings)]
            else:
                prefixes = [embeddings.to(device) for embeddings in imo_encoder(image)]
        
        prefix_embeds = [model.clip_project(prefix).reshape(1, prefix_length, -1) for prefix in prefixes]
        
    if use_beam_search:
        generated_text_prefix = [generate_beam(model, tokenizer, embed=prefix_embed)[0] for prefix_embed in prefix_embeds]
    else:
        generated_text_prefix = [generate2(model, tokenizer, embed=prefix_embed) for prefix_embed in prefix_embeds]
    
    print(f'\nUnnormalized: {generated_text_prefix[0]}')
    print(f'\nNormalized: {generated_text_prefix[1]}')
