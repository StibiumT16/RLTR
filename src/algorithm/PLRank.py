import torch
from torch.distributions.gumbel import Gumbel
from .base_algo import BaseRLAlgo
import utils.fair as fair
import utils.click_model as clm

# Computationally Efficient Optimization of Plackett-Luce Ranking Models for Relevance and Fairness
# For simplicity, we implement PL-Rank with Eq.(14) in the original paper, rather than more complicated Eq.(20) or Eq.(23). 
# All 3 equations estimate the same gradient.

class PLRank(BaseRLAlgo):
    def __init__(self, optimizer, scheduler, config, device, args):
        super().__init__(config, optimizer, scheduler, args)
        #super().__init__(optimizer, scheduler, args)
        #self.n_samples = config['algorithm'].get('sample', 8)
        #self.max_label = config['data']['max_label']
        
        #self.reward_func = config['algorithm'].get('reward', 'ndcg')
        
    
    def fit(self, model, input_data, device):
        model.train()
        
        label = input_data['label'].to(device) #[bs, k]
        output = model(input_data['feature'].to(device)).squeeze(-1) #[bs, k]
        batch_size, rank_list_size = label.shape
        
        gumbel_samples = Gumbel(loc = torch.ones((batch_size, self.n_samples, rank_list_size), device=device), scale = 1.0).sample()
        gumbel_scores = (gumbel_samples + output[:, None, :]).detach()
        _, ranking = torch.sort(gumbel_scores, dim = -1, descending = True) 
        
        ranking = ranking.view(batch_size * self.n_samples, -1)
        rank_score = torch.cat([output for _ in range(self.n_samples)], \
                dim = 1).view(batch_size * self.n_samples, -1) 
        
        __, conditional_log_pi = self.log_pi_batch(rank_score, ranking)
        
        with torch.no_grad():
            discount = (1. / torch.log2(torch.arange(rank_list_size, device=device) + 2.0) \
                ).repeat(self.n_samples * batch_size).view(self.n_samples * batch_size, -1)
            rank_label = torch.cat([label for _ in range(self.n_samples)], \
                dim = 1).view(batch_size * self.n_samples, -1) 
            suffix_reward = 0.
            
            for (reward_func, reward_weight) in zip(self.reward_funcs, self.reward_weights): 
                
                if reward_func == 'ndcg':
                    R = torch.gather(rank_label, dim = 1, index = ranking)
                    cur_suffix_reward = torch.flip(torch.cumsum(torch.flip(R * discount, dims= [1]), dim = 1), dims = [1])
                elif reward_func == 'click':
                    clicks = clm.click_simulation(ranking, rank_label, self.max_label, device, label.shape[1])
                    cur_suffix_reward = torch.flip(torch.cumsum(torch.flip(clicks * discount, dims= [1]), dim = 1), dims = [1])    
                elif reward_func == 'fair':
                    measure, __ = fair.fairness_measure(output, label, self.max_label, device)
                    ranking_measure = torch.cat([measure for _ in range(self.n_samples)], \
                        dim = 1).view(batch_size * self.n_samples, -1) 
                    ranking_measure = torch.gather(ranking_measure, dim = 1, index = ranking) # ?
                    cur_suffix_reward = torch.flip(torch.cumsum(torch.flip(ranking_measure * discount, dims= [1]), dim = 1), dims = [1])
                else:
                    raise NotImplementedError
                
                suffix_reward += reward_weight * cur_suffix_reward
                
            
        loss = -torch.sum(conditional_log_pi * suffix_reward, dim=1).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        model.zero_grad()
        
        return loss.item(), None
    
    