import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np

class temp_layer(pl.LightningModule):
    def __init__(self, temperature):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 /temperature))
