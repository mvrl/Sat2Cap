import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter 
import code

#local import
#from .geoclip import GeoClip

class Retrieval(object):
    def __init__(self, k):
        self.k = k
    
    def get_similarity(self):
        return  torch.matmul(self.normalized_overhead_img_embeddings,self.normalized_ground_img_embeddings.t()).T # (BxB - ground_imgs X overhead_imgs)


    def fit_k_similar(self, normalized_overhead_img_embeddings, normalized_ground_img_embeddings):
       
        self.normalized_overhead_img_embeddings = normalized_overhead_img_embeddings
        self.normalized_ground_img_embeddings = normalized_ground_img_embeddings

        self.similarity_per_ground_img = self.get_similarity()
        
        #get top k similar overhead img for each ground level img
        top_k_vals, top_k_idx = self.similarity_per_ground_img.topk(self.k)
        k_mask = torch.zeros_like(self.similarity_per_ground_img)
        k_mask.scatter_(1, top_k_idx, 1)
        #similarity matrix where top k matches are non zero are rest are all zero
        k_similar_matrix = k_mask*self.similarity_per_ground_img
        
        num_images = self.similarity_per_ground_img.shape[0]
        #total number of correct matches
        num_correct_predictions = torch.count_nonzero(torch.diagonal(k_similar_matrix))
        retrieval_metric = num_correct_predictions/num_images

        #save the k_similar_matrix
        self.k_similar_matrix = k_similar_matrix

        return retrieval_metric
        

def get_args():
    parser = ArgumentParser(description='arguments for runnning retrieval metrics', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--ckpt_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/GeoClip/or5comrl/checkpoints/epoch=21-step=15510.ckpt')
    parser.add_argument('--test_path', type=str, default='/home/a.dhakal/active/datasets/YFCC100m/webdataset/9f248448-1d13-43cb-a336-a7d92bc5359e.tar')
    parser.add_argument('--batch_size', type=int, default=512)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint['hyper_parameters']
    
    #set new hyper parameters
    hparams['batch_size'] = args.batch_size
    hparams['test_path'] = args.test_path    

    #no gradient context manager for evaluation
    with torch.set_grad_enabled(False):
        #load pretrained weights
        geoclip = GeoClip(hparams=hparams).eval()
        geoclip.load_state_dict(checkpoint['state_dict'])
        #fetch the test dataloader
        test_dataloader = geoclip.test_dataloader()
        sample = next(iter(test_dataloader))
        embeddings = geoclip(sample)
        print('Output keys:',embeddings.keys())
        ground_img_embeddings = embeddings['ground_img_embeddings']
        overhead_img_embeddings = embeddings['overhead_img_embeddings']
        keys = embeddings['keys']
        retrieval = Retrieval(k=5)
        metric = retrieval.fit_k_similar(overhead_img_embeddings, ground_img_embeddings)
        print(metric)
    code.interact(local=dict(globals(), **locals()))

# with torch.set_grad_enabled(False):
#     sample = next(iter(test_dataloader))
#     embeddings = geoclip(sample)
#     print('Output keys:',embeddings.keys())
#     ground_img_embeddings = embeddings['ground_img_embeddings']
#     overhead_img_embeddings = embeddings['overhead_img_embeddings']
#     keys = embeddings['keys']
#     retrieval = Retrieval(k=10)
#     metric = retrieval.fit_k_similar(overhead_img_embeddings, ground_img_embeddings)
#     print(metric)

    