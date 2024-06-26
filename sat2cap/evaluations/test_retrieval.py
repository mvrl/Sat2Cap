import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter, Namespace 
import code
from PIL import Image
from torchvision import transforms
#local imports
#from ..multidata_2 import MultiData
from ..multidata import MultiData as MultiData
from ..multidata_raw import MultiDataRaw
from ..models.geomoco import GeoMoCo
from ..utils import utils, preprocess
from .metrics import Retrieval


def get_args():
    parser = ArgumentParser(description='arguments for runnning retrieval metrics', formatter_class=RawTextHelpFormatter)

    #parser.add_argument('--ckpt_path', type=str, default='root_path/logs/GeoClip/u3oyk5ft/checkpoints/step=8600-val_loss=5.672.ckpt')
    #parser.add_argument('--ckpt_path', type=str, default='root_path/logs/temp_models/f1dtv48z/checkpoints/step=71250-val_loss=4.357.ckpt')
    parser.add_argument('--ckpt_path', type=str, default='root_path/logs/temp_models/s212e5he/checkpoints/step=38000-val_loss=4.957.ckpt')
    parser.add_argument('--test_path', type=str, default='data_dir/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--run_topk', action='store_true', default=False)
    parser.add_argument('--save_topk', action='store_true', default=False)
    parser.add_argument('--k', type=list, default=[5,10])
    parser.add_argument('--clip', action='store_true', help='Run inference on normal CLIP model')
    parser.add_argument('--geo_encode', action='store_true', default=False)
    parser.add_argument('--date_time', type=str, default='')
    parser.add_argument('--save_path', type=str, default='root_path/logs/evaluations/wacv/retrieval_images')
    parser.add_argument('--evaluation_dir', type=str, default='root_path/logs/evaluations/wacv/retrieval_results_node.txt')
    parser.add_argument('--g2o', action='store_true')
    args = parser.parse_args()
    return args


def get_retrieval_metric(model, sample, ks=[5,10], g2o=False):
    
    data_size = len(sample)
    embeddings = model(sample)
    print('Output keys:',embeddings.keys())
    ground_img_embeddings = embeddings['ground_img_embeddings']
    overhead_img_embeddings = embeddings['overhead_img_embeddings']
    print(f'Size of retrieval data {len(ground_img_embeddings)}')
    keys = embeddings['keys']
    metric = {}
    for k in ks:
        retrieval = Retrieval(k=k)
        curr_metric = retrieval.fit_k_similar(overhead_img_embeddings, ground_img_embeddings, g2o=g2o)
        metric[k] = curr_metric
        median_metric = retrieval.get_median_metric()
    
    return metric, median_metric


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def get_similarity(normalized_overhead_img_embeddings,normalized_ground_img_embeddings):
    return  torch.matmul(normalized_overhead_img_embeddings,normalized_ground_img_embeddings.t()) # (BxB - ground_imgs X overhead_imgs)

def de_norm(x,normalization):
    mean, std = normalization
    z = x * torch.tensor(std).view(3, 1, 1)
    z = z + torch.tensor(mean).view(3, 1, 1)
    return z

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def save_top_k(model, batch,args, k=5):
    # valid_imo_transforms = transforms.Compose([
    #         transforms.Resize((224,224)),
    #         transforms.CenterCrop(size=(224,224)),
    #         _convert_image_to_rgb,
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.3670, 0.3827, 0.3338), (0.2209, 0.1975, 0.1988))
    #     ])

    # img_normalize = [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]]
    # imo_normalize = [[0.3670, 0.3827, 0.3338], [0.2209, 0.1975, 0.1988]]
    # to_pil = transforms.ToPILImage(mode='RGB')
    geo_processor = preprocess.Preprocess()
    data_size = len(batch)

    ground_images_raw, overhead_images_raw, metadata, keys = batch
    
    overhead_images = geo_processor.preprocess_overhead(overhead_images_raw)
    ground_images = geo_processor.preprocess_ground(ground_images_raw)
    
    
    if not args.date_time:
        print('Using Ground Date_Time')
        geoencode = utils.get_stacked_geoencode(metadata)
    else:
        print('Using Custom Date-Time')
        new_meta = []
        for meta in metadata:
            temp_dict = {}
            temp_dict['latitude'] = meta['latitude']
            temp_dict['longitude'] = meta['longitude']
            temp_dict['date_taken'] = args.date_time
            new_meta.append(temp_dict)
        geoencode = utils.get_stacked_geoencode(new_meta)
            

    # overhead_images_new = torch.stack([valid_imo_transforms(img) for img in overhead_images_old])
    # print(overhead_images_new.shape)
    # ground_images = [to_pil(de_norm(im, img_normalize)) for im in ground_images_old]
    # overhead_images = overhead_images_old

    batch = (ground_images, overhead_images, geoencode, metadata, keys)
        
    with torch.no_grad():
        embeddings = model(batch)
    print('Output keys:',embeddings.keys())

    # Define the sample data
    ground_images_raw = [im.resize((224,224)) for im in ground_images_raw]
    sample = {'overhead_images': overhead_images_raw, 'ground_images': ground_images_raw, 'metadata': metadata}


    # Define a dictionary to map the sample ID to an integer index
    sample_index = {num: i for i, num in enumerate(embeddings['keys'])}

    # Convert the embeddings to PyTorch tensors
    overhead_embs = embeddings['overhead_img_embeddings']
    ground_embs = embeddings['ground_img_embeddings']

    # Compute the pairwise similarity between overhead and ground embeddings
    similarity_matrix = get_similarity(overhead_embs, ground_embs)

    # Find the top 5 closest embeddings for each overhead embedding
    top_k = k
    _, indices = torch.topk(similarity_matrix, k=top_k, dim=1)
    
    # Convert to numpy array for easier indexing
    indices = indices.to('cpu').numpy()  
    similarity_matrix = similarity_matrix.to('cpu').numpy()
    # Find the indices of the overhead embeddings where the correct ground embedding was among the top 5 closest
    correct_indices = []
    for i, sample_id in enumerate(embeddings['keys']):
        sample_idx = sample_index[sample_id]
        correct_idx = np.where(indices[i] == sample_idx)[0]
        if correct_idx.size > 0:
            correct_indices.append(i)

    # Save the top 10 overhead images with highest accuracy and their top 5 similar ground images
    top_k_indices = np.random.permutation(correct_indices)[0:50]
    top_k_indices = [i for i in range(300)]
    for i in top_k_indices:
        overhead_img = sample['overhead_images'][i]
        overhead_emb = embeddings['overhead_img_embeddings'][i].to('cpu')

        #get metadata for the overhead image
        curr_meta = metadata[i]
        curr_lat = curr_meta['latitude']
        curr_long = curr_meta['longitude']
        curr_key = keys[i]
        # Get the top 5 most similar ground level images and their corresponding similarities
        similarity_row = similarity_matrix[i]
        #similarity_row[sample_index[embeddings['keys'][i]]] = -1.0  # Remove the correct ground embedding from consideration
        top_ground_indices = np.argsort(similarity_row)[-top_k:]
        top_ground_embs = ground_embs[top_ground_indices]
        top_ground_sims = similarity_row[top_ground_indices]
        # Save the images to file, display them, or do any other desired operation
        print(f"Overhead image {i+1}:")

        print(f"Similarities are {top_ground_sims}")
        top_ground_imgs = [sample['ground_images'][j] for j in reversed(top_ground_indices)]
        #code.interact(local=dict(globals(), **locals()))

        ground_img_grid = image_grid(top_ground_imgs, 3,3)
        
        combined_img = image_grid([overhead_img, ground_img_grid], 1,2)
        
        r = np.random.randint(0,10000)
        combined_img.save(f'{args.save_path}/{curr_key}_lat_{curr_lat}_long_{curr_long}.jpg')
        #ground_img_grid.save(f'root_path/logs/evaluations/wacv/retrieval_images/ground_lat_{curr_lat}_long_{curr_long}_{r}.jpg')
        # for j, idx in enumerate(top_ground_indices):
        #     ground_img = sample['ground_images'][idx]
        #     ground_emb = embeddings['ground_img_embeddings'][idx].to('cpu').numpy()
        #     similarity = top_ground_sims[j]
        #     # Save the ground image
        #     # ...
        #     print(f"  {j+1}. Similarity: {similarity:.4f}")

def get_dataloader_1(val_batch_size, vali_path):
    loader_args = Namespace()
    loader_args.val_batch_size = val_batch_size
    loader_args.vali_path = vali_path
    dataset = MultiData(loader_args).get_ds('test')
    return dataset

def get_dataloader_raw(val_batch_size, vali_path):
    loader_args = Namespace()
    loader_args.val_batch_size = val_batch_size
    loader_args.vali_path = vali_path
    dataset = MultiDataRaw(loader_args).get_ds('test')
    return dataset

if __name__ == '__main__':
    utils.set_seed(56)
    args = get_args() 
    device = torch.device(args.device)
    #no gradient context manager for evaluation
    
    
    
    with open(args.evaluation_dir, 'a') as f:
        to_write = 'Writing for new experiment'
        f.write(to_write)
        f.write('_______________________________________________________\n')
    
    with torch.set_grad_enabled(False):
        #load pretrained weights
            checkpoint = torch.load(args.ckpt_path)
            hparams = checkpoint['hyper_parameters']
    
            #set new hyper parameters
            hparams['val_batch_size'] = args.batch_size
            hparams['test_path'] = args.test_path 
            hparams['prompt_loss'] = None         

            if not args.geo_encode:
                hparams['geo_encode'] = False
            print(f'Geo Encoding Used:{hparams["geo_encode"]}')      
            geoclip = GeoMoCo(hparams=hparams).eval().to(device)

            #set requires grad to false
            for param in geoclip.parameters():
                param.requires_grad=False
            if not args.clip:
                print('Using GeoClip')
                unused_params = geoclip.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f'Unused params {unused_params}')
            else:
                print('Using Regular CLIP')
            

            
            ######## code for model interpolation ##########################################################
            # #compute the metric for baseline CLIP
            # baseline_clip = GeoMoCo(hparams=hparams).eval().to(device)
            # for param in baseline_clip.parameters():
            #     param.requires_grad=False

            # #get the interpolated model
            # baseline_dict = baseline_clip.state_dict()
            # trained_dict = geoclip.state_dict()

            # for key in baseline_dict:
            #     trained_dict[key] = (1-args.geoclip_wt)*baseline_dict[key]+args.geoclip_wt*trained_dict[key]

            # interpolated_model = GeoClip(hparams=hparams).eval().to(device)
            # interpolated_model = interpolated_model
            # unused_params = interpolated_model.load_state_dict(trained_dict, strict=False)
            # print(f'Unused params {unused_params}')
            # for param in interpolated_model.parameters():
            #     param.requires_grad=False

            ##########################################################################################################################
        
            if args.run_topk:
                val_dataloader = get_dataloader_1(args.batch_size, args.test_path)
                sample = next(iter(val_dataloader))
                print('Samples Loaded')
                print('Running topk metric')
                geoclip_metric, median_metric = get_retrieval_metric(geoclip, sample, args.k, args.g2o)
                
                for key,val in geoclip_metric.items():
                    print(f'The top-{key} metric for geoclip model is {val}')
                print(f'The median metric for geoclip model is {median_metric}')
                
                with open(args.evaluation_dir, 'a') as f:
                    to_write = f'model:{args.ckpt_path}\ttop_k:{args.k}\tmetric:{geoclip_metric}\tmedian_metric:{median_metric}\tbatch_size:{args.batch_size}\tgeoencode:{args.geo_encode}\n'
                    f.write(to_write)
                    f.write('_______________________________________________________\n')

            if args.save_topk:
                val_dataloader = get_dataloader_raw(args.batch_size, args.test_path)
                sample = next(iter(val_dataloader))
                print('Samples Loaded')
                print('Saving topk')
                save_top_k(geoclip, sample,args, 9)
    # code.interact(local=dict(globals(), **locals()))
