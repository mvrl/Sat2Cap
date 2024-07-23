import numpy as np
import h5py
from argparse import ArgumentParser
import pandas as pd
import code
import os
#huggingface imports
from transformers import AutoTokenizer, CLIPTextModelWithProjection

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/home/a.dhakal/active/user_a.dhakal/geoclip/logs/geoclip_embeddings/usa/dropout/2012-05-20_08-00-00/dynamic_step=86750-val_loss=4.h5')
    parser.add_argument('--input_prompt', type=str, default='a photo of popular tourist destination')
    parser.add_argument('--date_time', type=str, default='2012-05-20 08:00:00.0')
    parser.add_argument('--vit_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--clip', action='store_true')

    args = parser.parse_args()
    return args

class TextSim():
    def __init__(self,vit_model):
        self.vit_model = vit_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.vit_model)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(self.vit_model)
    
    def get_text_embed(self, query):
        processed_query = self.tokenizer(query, padding=True, return_tensors='pt')
        text_outputs = self.text_model(**processed_query)
        text_embeddings = text_outputs.text_embeds.detach()
        return text_embeddings
    
    def get_top_loc(csv, k):
        df = pd.DataFrame(csv)
        df = df.sort_values(by='similarity')
        top_locations = df.iloc[:k,:]
        return top_locations


if __name__ == '__main__':
    args = get_args()
    
    #compute CLIP text embeddings
    
    textsim = TextSim(vit_model=args.vit_model)

    query = [args.input_prompt]
    text_embeddings = textsim.get_text_embed(query)

    #normalize the text embeddings
    normalized_text_embeddings = text_embeddings/text_embeddings.norm(p=2,dim=-1,keepdim=True)
    normalized_text_embeddings = normalized_text_embeddings.numpy()

    with h5py.File(args.input_path, 'r') as handle:
        print(handle.keys())
        
        #extract metadata of file
        # input_region = handle.attrs['input_region'].split('/')[-1]
        
        if args.clip:
            model_path = handle.attrs['model_path']
            overhead_embeddings = handle['overhead_embeddings']
            locations = handle['location']
            lat = locations[:,0]
            lon = locations[:,1]

            #normalized overhead embeddings
            normalized_overhead_embeddings = overhead_embeddings/np.linalg.norm(overhead_embeddings, axis=-1, keepdims=True, ord=2)

            #compute clip similarity
            similarities = normalized_overhead_embeddings @ normalized_text_embeddings.T
        else:    
            # model_path = handle.attrs['input_file_path']
            # date_time = handle.attrs['date_time']
        
            #extract data from file
            dynamic_embeddings = handle['dynamic_embeddings']
            locations = handle['location']
            lat = locations[:,0]
            lon = locations[:,1]

            #compute normalize dynamic embeddings
            normalized_dynamic_embeddings = dynamic_embeddings/np.linalg.norm(dynamic_embeddings,axis=-1,keepdims=True, ord=2)

            #compute text and location similarity
            similarities = normalized_dynamic_embeddings @ normalized_text_embeddings.T
        
        #save results to csv

    #get the output directory which is same as the input path directory  
    output_dir = os.path.dirname(args.input_path)

    #change input prompt name 
    modified_input_prompt = args.input_prompt.replace(' ','_')
    modified_input_prompt = f'normalized_{modified_input_prompt}'
    similarities = np.squeeze(similarities)
    df = pd.DataFrame({'lat':lat,'lon':lon, 'similarity':similarities})
    min_value = df["similarity"].min()
    max_value = df["similarity"].max()

    # Normalize the "similarity" column values between 0 and 1 using Min-Max normalization
    df["norm_similarity"] = (df["similarity"] - min_value) / (max_value - min_value)
    df.to_csv(f'{output_dir}/{modified_input_prompt}.csv')

   

        
        