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
from itertools import islice

ds = wds.WebDataset("data_dir/CVUSA/ted_subset/streetview.tar.gz")

sample = next(iter(ds))
print(sample.keys())
#code.interact(local=dict(globals(), **locals()))
# for sample in islice(ds, 0, 3):
#     for key, value in sample.items():
#         print(key, repr(value)[:50])
#     print()