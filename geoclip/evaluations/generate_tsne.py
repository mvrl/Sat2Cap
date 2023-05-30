import h5py
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import code



file_path = '/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/clip_embeddings/3fc3d5d7-d373-455f-8430-334217190f1c.h5'  # Replace with the actual path to your .h5 file
with h5py.File(file_path, 'r') as file:
    locations = file['location'][:]
    tensors = file['tensor'][:]

# Perform t-SNE dimensionality reduction
tensors = tensors[0:10000]
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_tsne = tsne.fit_transform(tensors)

# Plot the t-SNE results
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=4)
plt.title('t-SNE Plot of Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('/home/a.dhakal/active/user_a.dhakal/geoclip/logs/evaluations/wacv/tsne/tsne_clip.jpg')

#code.interact(local=dict(globals(), **locals()))
