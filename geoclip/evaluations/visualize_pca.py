import numpy as np
import h5py
import code
import matplotlib.pyplot as plt

def random_feature_viz(feats, locs):
    out_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/pca_visualizations'
    idx = 10
    samp = np.random.randint(0, locs.shape[0]-1, 1000)
    plt.scatter(locs[samp,1], locs[samp,0], c=feats[samp,idx])
    img_path = f'{out_path}/random_pca.jpeg'
    plt.savefig(img_path)
    plt.show()

def pca_viz(feats, locs):
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
    img_path = f'{out_path}/pca.jpeg'
    plt.figure(figsize=[20, 8])
    samp = np.random.randint(0, locs.shape[0]-1, 80000)
    plt.scatter(locs[samp,1], locs[samp,0], c=feats_clip[samp,:], marker=".");
    plt.savefig(img_path)
    plt.show()
    

if __name__ == '__main__':
    embedding_file = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/geoclip_embeddings/75d2f94b-d1e7-45c4-bed6-1b18db907dbb.h5'
    handle = h5py.File(embedding_file, 'r')
    print(f'The keys are {handle.keys()}')
    feats = handle['tensor'][:]
    locs = handle['location'][:]
    handle.close()
    pca_viz(feats,locs)
    #code.interact(local=dict(globals(), **locals()))