import torch
import torch.nn as nn
from .model_utils import * 

class DLCM(nn.Module):
    def __init__(self, params, feature_size, **kwargs):
        super(DLCM, self).__init__()
        
        hidden_layers = params.get('hidden_layers', [256, 128])
        expand_size = params.get('expand_size', 50)
        rnn_hidden_size = params.get('rnn_hidden_size', 128)
        rnn_num_layers = params.get('rnn_num_layers', 1)
        dropout = params.get('dropout', 0.)
        norm_func = params.get('norm_func', 'layer')
        act_func = params.get('act_func', 'relu')
        initializer = params.get('initializer', None)
        

        # Step 1: Feature abstraction
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module(f'linear0', nn.Linear(feature_size, (feature_size + expand_size) // 2))
        if norm_func:
            self.feature_extractor.add_module(f'norm0', norm_func_dict[norm_func]((feature_size + expand_size) // 2))
        if act_func:
            self.feature_extractor.add_module(f'act0', act_func_dict[act_func])
        self.feature_extractor.add_module(f'linear1', nn.Linear((feature_size + expand_size) // 2, expand_size))
            
        
        # Step 2: GRU Encoder for top documents
        self.rnn = nn.GRU(
            input_size=feature_size + expand_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout= 0 if rnn_num_layers == 1 else dropout
        )
        
        # Step 3: Re-ranking layer
        feature_size = rnn_hidden_size
        self.scorer = nn.Sequential()
        for i in range(len(hidden_layers)):
            self.scorer.add_module(f'linear{i}', nn.Linear(feature_size, hidden_layers[i]))
            feature_size = hidden_layers[i]
            if norm_func:
                self.scorer.add_module(f'norm{i}', norm_func_dict[norm_func](feature_size))
            if act_func:
                self.scorer.add_module(f'act{i}', act_func_dict[act_func])
        self.scorer.add_module(f'linear{i+1}', nn.Linear(feature_size, 1))

        if initializer:
            initializer = initializer_dict[initializer]
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    initializer(layer.weight)

    def forward(self, input, **kwargs): 
        batch_size, list_size, feature_size = input.shape
    
        input_flat = input.to(dtype=torch.float32).view(-1, feature_size)
        abstract_features = self.feature_extractor(input_flat)  # [batch_size*list_size, new_feature_size]
        
        concat_features = torch.cat([input_flat, abstract_features], dim=-1).view(batch_size, list_size, -1)  # [batch_size, list_size, concat_dim]
        
        rnn_out, _ = self.rnn(concat_features)  # rnn_out: [batch_size, list_size, rnn_hidden_size]
        scores = self.scorer(rnn_out).squeeze(-1)   # [batch_size, list_size, 1]
        
        return scores
