import torch
import torch.nn as nn
from .model_utils import * 

class Linear(nn.Module):
    def __init__(self, params, feature_size, **kwargs):
        super(Linear, self).__init__()
        self.sequential = nn.Sequential()
        
        act_func = params.get('act_func', None)
        initializer = params.get('initializer', None)
        
        self.sequential.add_module(f'linear0', nn.Linear(feature_size, 1))
        
        if act_func:
            self.sequential.add_module(f'act0', act_func_dict[act_func])
        
        if initializer:
            initializer = initializer_dict[initializer]
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    initializer(layer.weight)
    
    def forward(self, input, **kwargs):
        return self.sequential(input.to(dtype=torch.float32))