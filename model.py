import math
import torch
import torch.nn as nn


def relative_time_encoding()



class malcat(nn.Transformer):
    def __init__(self, 
                 dim_model=512, 
                 n_head=8, # so 64 dim/head
                 n_encoder_layers=6, 
                 n_decoder_layers=6,):
        
        
        super(malcat, self).__init__(
            d_model=dim_model,
            nhead=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
        )
    

