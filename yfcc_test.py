import os
import matplotlib.pyplot as plt
import numpy as np
import webdataset as wds
import code
import imageio
import numpy as np
from skimage.transform import resize
import json
import code

ds = wds.WebDataset("/home/a.dhakal/active/datasets/YFCC100m/webdataset/0a912f85-6367-4df4-aafe-b48e6e1d2be4.tar")
ds = ds.shuffle(1000)
ds = ds.decode("rgb")
ds = ds.to_tuple("groundlevel.jpg", "overhead.jpg", "metadata.json")

num_img = None
x_dim = []
y_dim = []

for i,(img, imo, json) in enumerate(ds):

    # print(f'Img Groundshape {img.shape}')
    # print(f'Img Overhead shape {imo.shape}')
    x,y,c = img.shape
    x_dim.append(x)
    y_dim.append(y)

    # block = np.zeros((img.shape[0],20,3))
    # imo = resize(imo, (img.shape))
    # combined = np.hstack((img,block,imo))
    # combined = (combined*255).astype(np.uint8)
    # imageio.imsave(f'image_{i}.png', combined)
    if i==num_img:
        print(json)
        break
x_mean = np.mean(x_dim)
y_mean = np.mean(y_dim)
print(x_mean,y_mean)
#code.interact(local=dict(globals(), **locals()))
#imgageio.imsave('./img/')
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(img)
# ax[1].imshow(imo)
# print(json)

# plt.show()

#code.interact(local=dict(globals(), **locals()))