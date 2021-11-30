import math
import torch.nn as nn
from models.layers import clones, LayerNorm, SublayerConnection



class Encoder(nn.Module):
    """
    - Core encoder is a stack of N layers.
    - The output of each sub-layer is LayerNorm(x+Sublayer(x))
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class EncoderLayer(nn.Module):
    """
     - Encoder is made up of self-attn and feed forward.
     - Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the 
     second is a simple, position-wise fully connected feed- forward network.
     """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

