import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter 
import code
from PIL import Image
#local imports
from ..models.geoclip import GeoClip
from .metrics import Retrieval


def get_args():
    parser = ArgumentParser(description='arguments for runnning retrieval metrics', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/u3oyk5ft/checkpoints/step=8600-val_loss=5.672.ckpt')
    parser.add_argument('--test_path', type=str, default='/home/a.dhakal/active/datasets/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--geoclip_wt', type=float, default=0.5)
    parser.add_argument('--run_topk', action='store_true')
    parser.add_argument('--save_topk', action='store_true')
    args = parser.parse_args()
    return args


def get_retrieval_metric(model, sample, k=1):
    data_size = len(sample)
    embeddings = model(sample)
    print('Output keys:',embeddings.keys())
    ground_img_embeddings = embeddings['ground_img_embeddings']
    overhead_img_embeddings = embeddings['overhead_img_embeddings']
    print(f'Size of retrieval data {len(ground_img_embeddings)}')
    keys = embeddings['keys']
    retrieval = Retrieval(k=k)
    metric = retrieval.fit_k_similar(overhead_img_embeddings, ground_img_embeddings)
    return metric

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

def save_top_k(model, batch, k=5):
    data_size = len(batch)
    ground_images, overhead_images, metadata, _ = batch
    with torch.no_grad():
        embeddings = model(batch)
    print('Output keys:',embeddings.keys())

    # Define the sample data
    sample = {'overhead_images': overhead_images, 'ground_images': ground_images, 'metadata': metadata}

    
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
    top_k_indices = np.random.permutation(correct_indices)[0:10]
    for i in top_k_indices:
        overhead_img = sample['overhead_images'][i]
        overhead_emb = embeddings['overhead_img_embeddings'][i].to('cpu')

        #get metadata for the overhead image
        curr_meta = metadata[i]
        curr_lat = curr_meta['latitude']
        curr_long = curr_meta['longitude']
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
        
        ground_img_grid = image_grid(top_ground_imgs, 3,3)
        
        combined_img = image_grid([overhead_img, ground_img_grid], 1,2)
        
        r = np.random.randint(0,1000)
        combined_img.save(f'/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/retrieval_images/lat_{curr_lat}_long_{curr_long}_{r}.jpg')

        # for j, idx in enumerate(top_ground_indices):
        #     ground_img = sample['ground_images'][idx]
        #     ground_emb = embeddings['ground_img_embeddings'][idx].to('cpu').numpy()
        #     similarity = top_ground_sims[j]
        #     # Save the ground image
        #     # ...
        #     print(f"  {j+1}. Similarity: {similarity:.4f}")


def get_retrieval_metric(model, sample, k=1):
    data_size = len(sample)
    embeddings = model(sample)
    print('Output keys:',embeddings.keys())
    ground_img_embeddings = embeddings['ground_img_embeddings']
    overhead_img_embeddings = embeddings['overhead_img_embeddings']
    print(f'Size of retrieval data {len(ground_img_embeddings)}')
    keys = embeddings['keys']
    retrieval = Retrieval(k=k)
    metric = retrieval.fit_k_similar(overhead_img_embeddings, ground_img_embeddings)
    return metric

if __name__ == '__main__':
    args = get_args() 
    device = torch.device(args.device)
    #no gradient context manager for evaluation
    
    evaluation_dir = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/retrieval_results.txt'
    alphas = [1]
    
    with open(evaluation_dir, 'a') as f:
        to_write = f'alpha\tTop 5 accuracy\tLength of Test Set\n'
        f.write(to_write)
        f.write('_______________________________________________________\n')
    
    for alpha in alphas:
        args.geoclip_wt=alpha
        print(f'Using alpha {args.geoclip_wt}')
        with torch.set_grad_enabled(False):
            #load pretrained weights
                checkpoint = torch.load(args.ckpt_path)
                hparams = checkpoint['hyper_parameters']
        
                #set new hyper parameters
                hparams['val_batch_size'] = args.batch_size
                hparams['test_path'] = args.test_path 
                hparams['prompt_loss'] = None               
                geoclip = GeoClip(hparams=hparams).eval().to(device)

                #set requires grad to false
                for param in geoclip.parameters():
                    param.requires_grad=False
                unused_params = geoclip.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f'Unused params {unused_params}')
                #fetch the test dataloader
                val_dataloader = geoclip.val_dataloader()
                sample = next(iter(val_dataloader))

                #compute the metric for baseline CLIP
                baseline_clip = GeoClip(hparams=hparams).eval().to(device)
                for param in baseline_clip.parameters():
                    param.requires_grad=False

                #get the interpolated model
                baseline_dict = baseline_clip.state_dict()
                trained_dict = geoclip.state_dict()

                for key in baseline_dict:
                    trained_dict[key] = (1-args.geoclip_wt)*baseline_dict[key]+args.geoclip_wt*trained_dict[key]

                interpolated_model = GeoClip(hparams=hparams).eval().to(device)
                interpolated_model = interpolated_model
                unused_params = interpolated_model.load_state_dict(trained_dict, strict=False)
                print(f'Unused params {unused_params}')
                for param in interpolated_model.parameters():
                    param.requires_grad=False
            
                if args.run_topk:
                    interpolated_metric = get_retrieval_metric(interpolated_model, sample, 5)
                    print(f'The retrieval metric for interpolated model is {interpolated_metric}')
                    
                    with open(evaluation_dir, 'a') as f:
                        to_write = f'{args.geoclip_wt}\t{interpolated_metric}\t{args.batch_size}\n'
                        f.write(to_write)
                        f.write('_______________________________________________________\n')

                if args.save_topk:
                    save_top_k(interpolated_model, sample, 9)
    # code.interact(local=dict(globals(), **locals()))
