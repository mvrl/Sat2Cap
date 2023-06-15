import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter 
import code

#local import
#from .geoclip import GeoClip

class Retrieval(object):
    def __init__(self, k):
        self.k = k
    
    def get_similarity(self):
        return  torch.matmul(self.normalized_overhead_img_embeddings,self.normalized_ground_img_embeddings.t()) # (BxB - ground_imgs X overhead_imgs)

    def get_median_metric(self):
        cos_sim = self.similarity_per_overhead_img.detach().cpu().numpy()
        distance_matrix = cos_sim
        K = cos_sim.shape[0]
        # Evaluate Img2Sound
        results = []
        for i in list(range(K)):
            tmpdf = pd.DataFrame(dict(
                k_snd = i,
                dist = distance_matrix[:, i]
            )).set_index('k_snd')

            tmpdf['rank'] = tmpdf.dist.rank(ascending=False)
            res = dict(
                rank=tmpdf.iloc[i]['rank']
            )
            results.append(res)
        df = pd.DataFrame(results)
        median_rank = df['rank'].median()
        
        return median_rank

        


    def fit_k_similar(self, normalized_overhead_img_embeddings, normalized_ground_img_embeddings, g2o=False):
       
        self.normalized_overhead_img_embeddings = normalized_overhead_img_embeddings
        self.normalized_ground_img_embeddings = normalized_ground_img_embeddings

        self.similarity_per_overhead_img = self.get_similarity()

        if g2o:
            self.similarity_per_overhead_img = self.similarity_per_overhead_img.T
        
        #get top k similar overhead img for each ground level img
        top_k_vals, top_k_idx = self.similarity_per_overhead_img.topk(self.k)
        k_mask = torch.zeros_like(self.similarity_per_overhead_img)
        k_mask.scatter_(1, top_k_idx, 1)
        #similarity matrix where top k matches are non zero are rest are all zero
        k_similar_matrix = k_mask*self.similarity_per_overhead_img
        
        num_images = self.similarity_per_overhead_img.shape[0]
        #total number of correct matches
        num_correct_predictions = torch.count_nonzero(torch.diagonal(k_similar_matrix))
        retrieval_metric = num_correct_predictions/num_images

        #save the k_similar_matrix
        self.k_similar_matrix = k_similar_matrix

        return retrieval_metric
