import torch
import torch.nn.functional as F
from .base_algo import BasePointAlgo
from utils.click_model import click_simulation

class IPW(BasePointAlgo):
    def __init__(self, optimizer, scheduler, config, device, args):
        super().__init__(optimizer, scheduler, args)
        self.max_label = config['data']['max_label']
        self.k = args.k
        self.PropensityWeights = torch.log2(torch.arange(self.k, dtype=torch.float, device=device) + 2.0)
    
    def fit(self, model, input_data, device):
        model.train()
        label = input_data['label'].to(device)
        
        output = model(input_data['feature'].to(device)).squeeze(-1)
        output_sorted, ranking = torch.sort(output, dim = -1, descending = True) 
        
        with torch.no_grad():
            clicks = click_simulation(ranking, label, max_label=self.max_label, device=device)
            ipw_label = clicks * self.PropensityWeights
            ipw_label = torch.nan_to_num(ipw_label / torch.sum(ipw_label, 1, keepdim=True))
    
        loss = F.cross_entropy(output_sorted, ipw_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        model.zero_grad()
        
        return loss.item(), None