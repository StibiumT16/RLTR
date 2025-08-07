import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_algo import BasePointAlgo
from utils.click_model import click_simulation

class REM(BasePointAlgo):
    def __init__(self, optimizer, scheduler, config, device, args):
        super().__init__(optimizer, scheduler, args)
        self.max_label = config['data']['max_label']
        self.EM_step_size = config['algorithm'].get('EM_step_size', 0.05)
        
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.propensity = torch.ones(args.k, device=device, requires_grad=False) * 0.9

    def fit(self, model, input_data, device):
        model.train()
        label = input_data['label'].to(device)
        
        output = model(input_data['feature'].to(device)).squeeze(-1)
        sorted_output, ranking = torch.sort(output, dim = -1, descending = True) 
        
        with torch.no_grad():
            clicks = click_simulation(ranking, label, max_label=self.max_label, device=device)
            
        gamma = torch.sigmoid(sorted_output) # [B, K]
        p_e1_r0_c0 = self.propensity * (1 - gamma) / (1 - self.propensity * gamma)
        p_e0_r1_c0 = (1 - self.propensity) * gamma / (1 - self.propensity * gamma)
        p_r1 = clicks + (1 - clicks) * p_e0_r1_c0
        
        rank_labels = torch.ceil(p_r1 - torch.rand(p_r1.shape, device=device))
        
        loss = self.criterion(sorted_output, rank_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        model.zero_grad()
        
        with torch.no_grad():
            self.propensity = (1 - self.EM_step_size) * self.propensity + \
                self.EM_step_size * torch.mean(clicks + (1 - clicks) * p_e1_r0_c0, dim=0)
        
        return loss.item(), None