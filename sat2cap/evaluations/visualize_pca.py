import numpy as np
import h5py
import code
import matplotlib.pyplot as plt
import glob
import pandas as pd

def random_feature_viz(feats, locs):
    out_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/pca_visualizations'
    idx = 10
    samp = np.random.randint(0, locs.shape[0]-1, 1000)
    plt.scatter(locs[samp,1], locs[samp,0], c=feats[samp,idx])
    img_path = f'{out_path}/random_pca.jpeg'
    plt.savefig(img_path)
    plt.show()

def generate_df(feats, locs):
    df = pd.DataFrame({'red':feats[:,0], 'green':feats[:,1], 'blue':feats[:,2], 'lat':locs[:,0], 'long':locs[:,1]})
    return df

def pca_viz(feats, locs, get_df=True):
    # PCA
    out_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/pca_visualizations'
    feats_cent = feats - feats.mean(axis=0, keepdims=True)
    [U, S, Vh] = np.linalg.svd(feats_cent.T @ feats_cent, compute_uv=True);
    feats_vis = feats_cent@U[:,:3];

    # convert to viable RGB values
    feats_pct = np.percentile(feats_vis, [5,95], axis=0)
    feats_clip = (feats_vis - feats_pct[(0,),:]) / (feats_pct[(1,),:] - feats_pct[(0,),:])
    feats_clip = feats_clip.clip(0,1)


    #plot 50000 points
    img_path = f'{out_path}/data_distribution.jpeg'
    plt.figure(figsize=[20, 8])
    #samp = np.linspace(0,len(locs)-1, len(locs))
    #plt.scatter(locs[:,1], locs[:,0], c=feats_clip[:,0], marker=".");
    plt.scatter(locs[:,1], locs[:,0], marker=".");
    plt.savefig(img_path)
    plt.show()
    if get_df:
        df = generate_df(feats_clip, locs)
        return df
    

if __name__ == '__main__':
    
    files = glob.glob('/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/clip_embeddings/*.h5')
    for i,file in enumerate(files):
        try:
            handle = h5py.File(file, 'r')
            print(f'The keys are {handle.keys()}')
            temp_feats = handle['tensor'][:]
            temp_locs = handle['location'][:]
            if i == 0:
                feats = temp_feats.copy()
                locs = temp_locs.copy()
            else:
                feats = np.vstack([feats,temp_feats])
                locs = np.vstack([locs, temp_locs])    
            handle.close()
        except BlockingIOError:
            print('File currently in progress')
    print(f'Length of data is {len(locs)}')
 #   code.interact(local=dict(globals(), **locals()))
    df = pca_viz(feats,locs, get_df=False)
    