import torch
from torch.distributions.gumbel import Gumbel
from .metrics import _prepare


def fairness_measure(probs, labels, max_label, device, sample_times = 100):
    labels, K = _prepare(probs, labels, device, None)
    #K = labels.shape[1]
    B = labels.shape[0]
    
    gumbel_samples = Gumbel(loc = torch.ones((B, sample_times, K), device=device), scale = 1.0).sample()
    gumbel_scores = (gumbel_samples + probs[:, None, :]).detach()
    _, sampled_ranking = torch.sort(gumbel_scores, dim = -1, descending = True) 
    sampled_ranking = sampled_ranking.permute(1, 0, 2)
    
    E = torch.zeros_like(labels, dtype=float, device=device)
    cur_range = range(B)
    
    for sample in sampled_ranking:
        for i in range(K):
            E[cur_range, sample[:, i]] += 1 / torch.log2(torch.tensor(i + 2))           
    
    R = (2 ** labels - 1) / (2 ** max_label - 1)
    swap_reward = E[:, :,None] * R[:, None,:]
    disparity = swap_reward.permute(0, 2, 1) - swap_reward
    
    measure = torch.sum(disparity * R[:, None,:], dim = 2) * 4 / (K * (K - 1))    
    unfairness = torch.sum(disparity ** 2., dim = [1,2]) / (K * (K - 1))    
    
    return measure, unfairness



def fairness_metric(probs, labels, max_label, device,  sample_times = 100):
    __, unfair = fairness_measure(probs, labels,  max_label, device, sample_times = sample_times)
    return -unfair, -unfair.mean()



def fairness_reward(rankings, probs,  labels, max_label, n_samples, device, topn=None, sample_times = 100):
    B_N, K = rankings.shape  
    discount = (1. / torch.log2(torch.arange(K, device=device) + 2.0)).repeat(B_N).view(B_N, -1)
    
    measure, __  = fairness_measure(probs, labels,  max_label, device, sample_times = sample_times)
    rank_measure = torch.cat([measure for _ in range(n_samples)], dim = 1)
    rank_measure = rank_measure.view(B_N, -1) 
    rank_measure = torch.gather(rank_measure, dim = 1, index = rankings[:, :topn])
    
    fair_reward = torch.sum(rank_measure * discount, dim = 1)
    
    return fair_reward, fair_reward.mean()