import torch
import copy
from torch.distributions.gumbel import Gumbel
#import utils.dicts as dicts
from .base_algo import BaseRLAlgo
#import utils.fair as fair

class GRPO(BaseRLAlgo):
    def __init__(self, optimizer, scheduler, config, device, args):
        super().__init__(config, optimizer, scheduler, args)
        #super().__init__(optimizer, scheduler, args)
        #self.n_samples = config['algorithm'].get('group_size', 8)
        #self.reward_func = config['algorithm'].get('reward', 'ndcg@10')
        #self.max_label = config['data']['max_label']  
        self.batch_epoch = config['algorithm'].get('batch_epoch', 1)
        self.beta = config['algorithm'].get('beta', 0.04)
        self.eps = config['algorithm'].get('eps', 0.2)
        self.ref_model_update_step = config['algorithm'].get('rel_model_update_step', 500)
        
        self.step = 0
              
        #try:
        #    self.reward_topk = int(self.reward_func.split('@')[1])
        #except:
        #    self.reward_topk = None
            
        #self.reward_func = self.reward_func.split('@')[0]
    
    def fit(self, model, input_data, device):
        model.train()
        
        self.step += 1
        if self.step % self.ref_model_update_step == 1:
            self.ref_model = copy.deepcopy(model)
        
        label = input_data['label'].to(device) #[bs, k]
        input_feature = input_data['feature'].to(device)
        
        old_output = model(input_feature).squeeze(-1) #[bs, k]
        batch_size = old_output.shape[0]
        
        gumbel_samples = Gumbel(loc = torch.ones((batch_size, self.n_samples, label.shape[-1]), device=device), scale = 1.0).sample()
        gumbel_scores = (gumbel_samples + old_output[:, None, :]).detach()
        __, ranking = torch.sort(gumbel_scores, dim = -1, descending = True) 
        
        ranking = ranking.view(batch_size * self.n_samples, -1)
        old_rank_score = torch.cat([old_output for _ in range(self.n_samples)], \
                dim = 1).view(batch_size * self.n_samples, -1) 

        with torch.no_grad():
            '''
            if self.reward_func == 'fair':
                reward, avg_reward = fair.fairness_reward(ranking, old_output, label, self.max_label, self.n_samples, device, self.reward_topk)
            else:
                reward, avg_reward = dicts.metric_dict[self.reward_func](ranking, rank_label, self.max_label, device, self.reward_topk)
            '''
            reward, avg_reward = self.get_reward(ranking, old_output, label, device)
            
            reward = reward.view(batch_size, self.n_samples)
            advantage = (reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True, unbiased=False) + 1e-10) # reward normalization
            advantage = advantage.view(-1)
            
            old_log_pi, __ = self.log_pi_batch(old_rank_score, ranking)
            
            ref_output = self.ref_model(input_feature).squeeze(-1)
            ref_rank_score = torch.cat([ref_output for _ in range(self.n_samples)], \
                dim = 1).view(batch_size * self.n_samples, -1) 
            ref_log_pi, __ = self.log_pi_batch(ref_rank_score, ranking)
            
        
        avg_loss = 0
        
        if self.batch_epoch == 1:
            log_pi, __ = self.log_pi_batch(old_rank_score, ranking)
            coef = torch.exp(log_pi - old_log_pi)
            kl = torch.exp(ref_log_pi - log_pi) - (ref_log_pi - log_pi) - 1
            
            loss = -torch.mean(coef * advantage - self.beta * kl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            model.zero_grad()
            avg_loss = loss.item()
            
        else:
            for _ in range(self.batch_epoch):
                output = model(input_feature).squeeze(-1)
                rank_score = torch.cat([output for _ in range(self.n_samples)], \
                    dim = 1).view(batch_size * self.n_samples, -1) 
                log_pi, __ = self.log_pi_batch(rank_score, ranking)
                
                coef1 = torch.exp(log_pi - old_log_pi)
                coef2 = torch.clamp(coef1, 1 - self.eps, 1 + self.eps)
                kl = torch.exp(ref_log_pi - log_pi) - (ref_log_pi - log_pi) - 1
                
                loss = -torch.mean(torch.min(coef1 * advantage, coef2 * advantage) - self.beta * kl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                model.zero_grad()
                avg_loss += loss.item()
            
        self.scheduler.step()
        
        return avg_loss / self.batch_epoch, avg_reward.item()
