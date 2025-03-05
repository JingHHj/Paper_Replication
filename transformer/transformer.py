import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self,embed__size, heads):
        super(SelfAttention, self).__init__()
        self.embed__size = embed__size
        self.heads = heads
        self.head_dim = embed__size // heads    # interger division

        assert (self.head_dim * heads == embed__size), "embed__size must be divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed__size)
                                # cocacnate the result
    def forward(self, values, keys, query, mask):
        N = query.shape[0]    
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  
        # they would related to the source sentance length and targer sentence length

        # split embedding into self.head pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum()
        # quieries.shape: (N, query_len, heads, head_dim)
        # keys.shape: (N, key_len, heads, head_dim)
        # energy.shape: (N, heads, query_len, key_len)