import torch
from torch.distributions.gumbel import Gumbel
#import utils.dicts as dicts
from .base_algo import BaseRLAlgo
#import utils.fair as fair

class PGRank(BaseRLAlgo):
    def __init__(self, optimizer, scheduler, config, device, args):
        super().__init__(config, optimizer, scheduler, args)
        '''
        super().__init__(optimizer, scheduler, args)
        self.n_samples = config['algorithm'].get('sample', 8)
        self.reward_func = config['algorithm'].get('reward', 'ndcg@10')
        self.max_label = config['data']['max_label']
        
        try:
            self.reward_topk = int(self.reward_func.split('@')[1])
        except:
            self.reward_topk = None
        self.reward_func = self.reward_func.split('@')[0]
        '''
    
    def fit(self, model, input_data, device):
        model.train()
        
        label = input_data['label'].to(device) #[bs, k]
        output = model(input_data['feature'].to(device)).squeeze(-1) #[bs, k]
        batch_size = output.shape[0]

        gumbel_samples = Gumbel(loc = torch.ones((batch_size, self.n_samples, label.shape[-1]), device=device), scale = 1.0).sample()
        gumbel_scores = (gumbel_samples + output[:, None, :]).detach()
        _, ranking = torch.sort(gumbel_scores, dim = -1, descending = True) 
        
        ranking = ranking.view(batch_size * self.n_samples, -1)
        rank_score = torch.cat([output for _ in range(self.n_samples)], \
                dim = 1).view(batch_size * self.n_samples, -1) 
        
        with torch.no_grad():
            '''
            if self.reward_func == 'fair':
                reward, avg_reward = fair.fairness_reward(ranking, output, label, self.max_label, self.n_samples, device, self.reward_topk)
            else:
                reward, avg_reward = dicts.metric_dict[self.reward_func](ranking, rank_label, self.max_label, device, self.reward_topk)
            '''
            reward, avg_reward = self.get_reward(ranking, output, label, device)
            if self.n_samples > 1:
                reward = reward.view(batch_size, self.n_samples)
                reward = reward - reward.mean(dim=1, keepdim=True) # baseline
                reward = reward.view(-1)
            
            
        log_pi, __ = self.log_pi_batch(rank_score, ranking)
        loss = -torch.mean(reward * log_pi)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        model.zero_grad()
        
        return loss.item(), avg_reward.item()
    