import torch
import torch.nn.functional as F
from .base_algo import BasePointAlgo

class CrossEntropy(BasePointAlgo):
    def __init__(self, optimizer, scheduler, config, device, args):
        super().__init__(optimizer, scheduler, args)
    
    def fit(self, model, input_data, device):
        model.train()
        
        label = input_data['label'].to(device) #[bs, k]
        norm_label = torch.nan_to_num(label / torch.sum(label, 1, keepdim=True))
        output = model(input_data['feature'].to(device)).squeeze(-1) #[bs, k]
        loss = F.cross_entropy(output, norm_label)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        model.zero_grad()
        
        return loss.item(), None
    
    