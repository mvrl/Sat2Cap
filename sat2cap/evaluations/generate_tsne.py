import h5py as h5
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import code


# clip_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/clip_embeddings/3fc3d5d7-d373-455f-8430-334217190f1c.h5'
# geoclip_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/geoembed_embeddings/epoch=3-step=29500/3fc3d5d7-d373-455f-8430-334217190f1c.h5'
# file_path = clip_path  # Replace with the actual path to your .h5 file
# with h5py.File(file_path, 'r') as file:
#     locations = file['location'][:]
#     tensors = file['tensor'][:]

# # Perform t-SNE dimensionality reduction
# tensors = tensors[0:10000]
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=5000, metric='cosine')
# embeddings_tsne = tsne.fit_transform(tensors)

# # Plot the t-SNE results
# plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=4)
# plt.title('t-SNE Plot of Embeddings')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.savefig('/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/tsne/cosine_tsne_clip.jpg')

#code.interact(local=dict(globals(), **locals()))
if __name__ == '__main__':
    input_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/geoclip_embeddings/test_set/test_embeddings.h5'
    handle = h5.File(input_path, 'r')         
  #  code.interact(local=dict(globals(), **locals()))
    overhead_geoclip_embeddings = handle['overhead_geoclip_embeddings'][:10000]
    norm_overhead_geoclip_embeddings = overhead_geoclip_embeddings/np.linalg.norm(overhead_geoclip_embeddings,axis=-1,keepdims=True, ord=2)

    overhead_clip_embeddings = handle['overhead_clip_embeddings'][:10000]
    norm_overhead_clip_embeddings = overhead_clip_embeddings/np.linalg.norm(overhead_clip_embeddings,axis=-1,keepdims=True, ord=2)

    ground_clip_embeddings = handle['ground_clip_embeddings'][:10000]
    norm_ground_clip_embeddings = ground_clip_embeddings/np.linalg.norm(ground_clip_embeddings,axis=-1,keepdims=True, ord=2)    

    all_embeddings = np.concatenate((norm_overhead_geoclip_embeddings, norm_overhead_clip_embeddings, norm_ground_clip_embeddings), axis=0)

    labels = np.array([0] * norm_overhead_geoclip_embeddings.shape[0] + [1] * norm_overhead_clip_embeddings.shape[0] +[2]*norm_ground_clip_embeddings.shape[0])

    tsne = TSNE(n_components=2, random_state=42, perplexity=100, n_iter=1000, metric='cosine')
    embeddings_tsne = tsne.fit_transform(norm_overhead_clip_embeddings)

    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=4)
    plt.title('t-SNE Visualization of CLIP Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/tsne/tsne_geoclip.jpg', dpi=250)
    plt.show()

