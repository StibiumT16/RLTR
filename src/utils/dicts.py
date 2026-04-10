import torch
import data, model, algorithm
from .metrics import *
from .click_model import click_reward

model_dict = {
    'dnn' : model.DNN,
    'linear' : model.Linear,
    'dlcm' : model.DLCM,
    'setrank' : model.SetRank
}

algorithm_dict = {
    'crossentropy' : algorithm.CrossEntropy,
    'attentionrank' : algorithm.AttentionRank,
    'lambdarank' : algorithm.LambdaRank,
    'pgrank' : algorithm.PGRank,
    'plrank0' : algorithm.PLRank0,
    'plrank3' : algorithm.PLRank3,
    'ppg' : algorithm.PPG,
    'grpo' : algorithm.GRPO,
    'ipw' : algorithm.IPW,
    'rem' : algorithm.REM,
}

optimizer_dict = {
    'adagrad' : torch.optim.Adagrad,
    'sgd' : torch.optim.SGD,
    'adam' : torch.optim.Adam,
    'adamw' : torch.optim.AdamW,
}

metric_dict = {
    'dcg' : dcg,
    'ndcg' : ndcg,
    'err' : err,
    'click' : click_reward,
}

input_feed_dict = {
    'eval': data.eval_direct_label_input,
    'direct_label_input': data.direct_label_input,
    'deterministic_online_label_input': data.deterministic_online_label_input,
    'stochastic_online_label_input': data.stochastic_online_label_input,
}

