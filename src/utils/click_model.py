import torch
from .metrics import _prepare

def click_simulation(rankings, labels, max_label, device, topn = None, epsilon = 0.1):
    labels, topn = _prepare(rankings, labels, device, topn)
    position_bias = torch.tensor(1, device=device) / \
        torch.log2(torch.arange(topn, dtype=torch.float, device=device) + 2.0)
    ranking_labels = torch.gather(labels, dim=1, index=rankings)
    
    relevance = (torch.pow(torch.tensor(2.0, device=device), ranking_labels) - 1) / \
      (torch.pow(torch.tensor(2.0, device=device), torch.as_tensor(max_label,device=device)) - 1)
    relevance = epsilon + (1 - epsilon) * relevance
    
    simulation = torch.rand(relevance.shape, device=device)
    clicks = (simulation <= relevance * position_bias).to(torch.float)
    
    return clicks # [batch_size, topn]
    
def click_reward(rankings, labels, max_label, device, topn = None, epsilon = 0.1):
    simulated_click = click_simulation(rankings, labels, max_label, device, topn, epsilon)
    
    #mask = (simulated_click.cumsum(dim=1) == 1) & (simulated_click == 1)
    #indices = mask.float() * torch.arange(1, simulated_click.shape[1] + 1, device=device).float()
    #first_pos = indices.sum(dim=1)
    #first_pos_score = torch.where(first_pos > 0, 1.0 / torch.log2(first_pos + 1), torch.zeros_like(first_pos))
    
    click_reward = torch.sum(simulated_click, dim=1) #+ first_pos_score
    
    return click_reward, torch.mean(click_reward)