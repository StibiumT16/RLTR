import torch.nn as nn
import torch.nn.functional as F

act_func_dict = {
    'relu' : nn.ReLU(),
    'tanh' : nn.Tanh(),
    'elu'  : nn.ELU(),
    'gelu' : nn.GELU(),
}

norm_func_dict = {
    'layer' : nn.LayerNorm,
    'batch' : nn.BatchNorm2d,
}

initializer_dict = {
    'xavier' : nn.init.xavier_normal_,
    'kaiming' : nn.init.kaiming_normal_,
}