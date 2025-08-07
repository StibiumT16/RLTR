import torch
import torch.nn as nn
from .model_utils import * 

class SetRank(nn.Module):
    def __init__(self, params, feature_size, **kwargs):
        super(SetRank, self).__init__() 
        
        num_layers = params.get('num_layers', 2)
        d_model = params.get('d_model', 256)
        n_head = params.get('n_head', 8)
        act_func = params.get('act_func', 'relu')
        layer_norm_eps = params.get('layer_norm_eps', 1e-8)
        dropout = params.get('dropout', 0.)
        
        encoder_layer = nn.TransformerEncoderLayer( #disable position embedding
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=4*d_model, 
            dropout=dropout,
            activation=act_func,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        
        self.input_ffn = nn.Sequential(
            nn.Linear(feature_size, 4*d_model),
            act_func_dict[act_func],
            nn.LayerNorm(4*d_model),
            nn.Linear(4*d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_ffn = nn.Linear(d_model, 1)
        
    def forward(self, input, **kwargs):
        x = self.input_ffn(input.to(dtype=torch.float32))
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.output_ffn(x)
        return x