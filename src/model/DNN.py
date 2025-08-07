import torch
import torch.nn as nn
from .model_utils import * 

class DNN(nn.Module):
    def __init__(self, params, feature_size, **kwargs):
        super(DNN, self).__init__()
        self.sequential = nn.Sequential()
        
        hidden_layers = params.get('hidden_layers', [512,256,128])
        norm_func = params.get('norm_func', None)
        act_func = params.get('act_func', None)
        initializer = params.get('initializer', None)
        
        for i in range(len(hidden_layers)):        
            self.sequential.add_module(f'linear{i}', nn.Linear(feature_size, hidden_layers[i]))
            feature_size = hidden_layers[i]
            
            if norm_func:
                self.sequential.add_module(f'norm{i}', norm_func_dict[norm_func](feature_size))
            
            if act_func:
                self.sequential.add_module(f'act{i}', act_func_dict[act_func])
            
            
        self.sequential.add_module(f'linear{i+1}', nn.Linear(feature_size, 1))
        
        if initializer:
            initializer = initializer_dict[initializer]
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    initializer(layer.weight)
        
        
    def forward(self, input, **kwargs):
        return self.sequential(input.to(dtype=torch.float32))